from __future__ import annotations

import json
import platform
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from omnifuser.data import (
    CLASS_NAMES,
    leave_one_tool_out_folds,
    nonastreda_source_signature,
)
from omnifuser.spectrogram import SpectrogramSettings
from omnifuser.tracking import ExperimentTracker
from omnifuser.train import model_config, save_arrays, train_fold

RECORD_IMAGE = re.compile(r"^T(?P<tool>\d+)R\d+B\d+$")
EVALUATION_MODES = ("full", "sensor_missing", "image_missing")
EVALUATION_METRIC_PATHS = {
    "classification_accuracy": ("classification", "accuracy"),
    "classification_precision_macro": ("classification", "precision_macro"),
    "classification_recall_macro": ("classification", "recall_macro"),
    "classification_f1_macro": ("classification", "f1_macro"),
    "classification_precision_weighted": ("classification", "precision_weighted"),
    "classification_recall_weighted": ("classification", "recall_weighted"),
    "classification_f1_weighted": ("classification", "f1_weighted"),
    "classification_roc_auc_macro": ("classification", "roc_auc_ovr", "macro"),
    "regression_mae": ("regression", "overall", "mae"),
    "regression_mse": ("regression", "overall", "mse"),
    "regression_rmse": ("regression", "overall", "rmse"),
    "regression_mae_normalized": ("regression", "normalized", "mae"),
    "regression_mse_normalized": ("regression", "normalized", "mse"),
    "regression_rmse_normalized": ("regression", "normalized", "rmse"),
}


def discover_tools(data_dir: str | Path) -> list[str]:
    image_dir = Path(data_dir) / "tool"
    tools: set[str] = set()
    for path in image_dir.glob("*.jpg"):
        match = RECORD_IMAGE.fullmatch(path.stem)
        if match is not None:
            tools.add(f"T{int(match.group('tool'))}")
    ordered = sorted(tools, key=lambda tool: int(tool[1:]))
    if len(ordered) != 10:
        raise ValueError(f"10-fold tool CV requires 10 tools; found {len(ordered)} in {image_dir}")
    return ordered


def _summarize_folds(results: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_paths = {
        "classification_accuracy": ("test", "classification_accuracy"),
        "classification_precision_macro": ("test", "classification", "precision_macro"),
        "classification_recall_macro": ("test", "classification", "recall_macro"),
        "classification_f1_macro": ("test", "classification", "f1_macro"),
        "classification_precision_weighted": ("test", "classification", "precision_weighted"),
        "classification_recall_weighted": ("test", "classification", "recall_weighted"),
        "classification_f1_weighted": ("test", "classification", "f1_weighted"),
        "classification_roc_auc_macro": ("test", "classification", "roc_auc_ovr", "macro"),
        "force_mae": ("test", "regression", "overall", "mae"),
        "force_mse": ("test", "regression", "overall", "mse"),
        "force_rmse": ("test", "regression", "overall", "rmse"),
        "force_mae_normalized": ("test", "regression", "normalized", "mae"),
        "force_mse_normalized": ("test", "regression", "normalized", "mse"),
        "force_rmse_normalized": ("test", "regression", "normalized", "rmse"),
    }
    summary: dict[str, Any] = {"folds": len(results), "metrics": {}}
    for name, path in scalar_paths.items():
        per_fold: list[float | None] = []
        for result in results:
            value: Any = result
            try:
                for key in path:
                    value = value[key]
            except KeyError:
                value = None
            per_fold.append(None if value is None else float(value))
        values = [value for value in per_fold if value is not None]
        summary["metrics"][name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "per_fold": per_fold,
        }
    return summary


def _summarize_paper_horizons(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate cumulative normalized metrics and their horizon-matched baselines."""
    summary: dict[str, Any] = {"models": {}, "baselines": {}}
    horizons = sorted(
        {
            horizon
            for result in results
            for horizon in result["test"].get("paper_horizons", {})
        },
        key=int,
    )
    for horizon in horizons:
        rows = [
            result["test"]["paper_horizons"][horizon]
            for result in results
            if horizon in result["test"].get("paper_horizons", {})
        ]
        summary["models"][horizon] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in rows])),
                "std": float(np.std([row[metric] for row in rows])),
            }
            for metric in sorted(rows[0])
            if all(metric in row for row in rows)
        }
        baseline_rows = [
            result["test"]["baselines"]["by_horizon"][horizon]
            for result in results
            if horizon in result["test"].get("baselines", {}).get("by_horizon", {})
        ]
        if baseline_rows:
            summary["baselines"][horizon] = {
                metric: {
                    "mean": float(np.mean([row[metric] for row in baseline_rows])),
                    "std": float(np.std([row[metric] for row in baseline_rows])),
                }
                for metric in sorted(baseline_rows[0])
                if all(metric in row for row in baseline_rows)
            }
    return summary


def _summarize_modality_evaluations(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_paths = EVALUATION_METRIC_PATHS
    summary: dict[str, Any] = {}
    for mode in EVALUATION_MODES:
        summary[mode] = {}
        for stage in ("train", "validation", "test"):
            per_metric: dict[str, list[float | None]] = {
                name: [] for name in metric_paths
            }
            for result in results:
                try:
                    evaluation = result["modality_evaluations"][mode][stage]
                except KeyError:
                    evaluation = {}
                for name, path in metric_paths.items():
                    value: Any = evaluation
                    try:
                        for key in path:
                            value = value[key]
                    except KeyError:
                        value = None
                    per_metric[name].append(
                        None if value is None else float(value)
                    )
            summary[mode][stage] = {}
            for name, fold_values in per_metric.items():
                values = [value for value in fold_values if value is not None]
                summary[mode][stage][name] = {
                    "mean": float(np.mean(values)) if values else None,
                    "std": float(np.std(values)) if values else None,
                    "per_fold": fold_values,
                }
    return summary


def _aggregate_out_of_fold(
    results: list[dict[str, Any]], output_dir: Path, mode: str = "full"
) -> dict[str, Any]:
    """Pool held-out test predictions of every fold for one evaluation mode."""
    if mode not in EVALUATION_MODES:
        raise ValueError(f"unknown evaluation mode: {mode}")
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    regression_abs_error = 0.0
    regression_squared_error = 0.0
    regression_elements = 0
    normalized_abs_error = 0.0
    normalized_squared_error = 0.0
    normalized_elements = 0
    channel_abs_error = np.zeros(3, dtype=np.float64)
    channel_squared_error = np.zeros(3, dtype=np.float64)
    channel_elements = 0
    roc_grid = np.linspace(0.0, 1.0, 1001)
    roc_tprs: dict[str, list[np.ndarray]] = {name: [] for name in CLASS_NAMES}
    roc_fold_auc: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}

    for result in results:
        if mode == "full":
            test_metrics = result["test"]
            prediction_artifact = result["artifacts"]["test_predictions"]
            roc_artifact = result["artifacts"]["roc_curves"]
        else:
            test_metrics = result["modality_evaluations"][mode]["test"]
            prediction_artifact = result["artifacts"][f"{mode}_predictions"]
            roc_artifact = result["artifacts"][f"{mode}_roc_curves"]
        confusion += np.asarray(
            test_metrics["classification"]["confusion_matrix"], dtype=np.int64
        )
        prediction_path = output_dir / prediction_artifact
        with np.load(prediction_path) as predictions:
            error = predictions["force_predictions"] - predictions["force_targets"]
            regression_abs_error += float(np.abs(error).sum())
            regression_squared_error += float(np.square(error).sum())
            regression_elements += error.size
            channel_abs_error += np.abs(error).sum(axis=(0, 1))
            channel_squared_error += np.square(error).sum(axis=(0, 1))
            channel_elements += error.shape[0] * error.shape[1]
            if "normalizer_scale" in result:
                # Each fold is normalized with its own training-tool statistics.
                scale = np.asarray(result["normalizer_scale"], dtype=np.float64)
                normalized = error / scale[None, None, :]
                normalized_abs_error += float(np.abs(normalized).sum())
                normalized_squared_error += float(np.square(normalized).sum())
                normalized_elements += normalized.size

        roc_path = output_dir / roc_artifact
        with np.load(roc_path) as curves:
            for name in CLASS_NAMES:
                auc = test_metrics["classification"]["roc_auc_ovr"]["per_class"][name]
                if auc is None:
                    continue
                fpr = curves[f"{name}_fpr"]
                tpr = curves[f"{name}_tpr"]
                roc_tprs[name].append(np.interp(roc_grid, fpr, tpr))
                roc_fold_auc[name].append(float(auc))

    true_positive = np.diag(confusion)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    support = confusion.sum(axis=1) / max(confusion.sum(), 1)
    class_report = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(confusion[index].sum()),
        }
        for index, name in enumerate(CLASS_NAMES)
    }
    aggregate: dict[str, Any] = {
        "classification": {
            "accuracy": float(true_positive.sum() / max(confusion.sum(), 1)),
            "precision_macro": float(precision.mean()),
            "recall_macro": float(recall.mean()),
            "f1_macro": float(f1.mean()),
            "precision_weighted": float((support * precision).sum()),
            "recall_weighted": float((support * recall).sum()),
            "f1_weighted": float((support * f1).sum()),
            "confusion_matrix": confusion.tolist(),
            "per_class": class_report,
            "roc_auc_ovr_macro_fold_mean": {
                name: {
                    "mean": float(np.mean(roc_fold_auc[name]))
                    if roc_fold_auc[name]
                    else None,
                    "std": float(np.std(roc_fold_auc[name]))
                    if roc_fold_auc[name]
                    else None,
                }
                for name in CLASS_NAMES
            },
        },
        "regression": {
            "units": "original force units",
            "overall": {
                "mae": regression_abs_error / max(regression_elements, 1),
                "mse": regression_squared_error / max(regression_elements, 1),
                "rmse": float(np.sqrt(regression_squared_error / max(regression_elements, 1))),
            },
            "by_channel": {
                channel: {
                    "mae": float(channel_abs_error[index] / max(channel_elements, 1)),
                    "mse": float(channel_squared_error[index] / max(channel_elements, 1)),
                    "rmse": float(np.sqrt(channel_squared_error[index] / max(channel_elements, 1))),
                }
                for index, channel in enumerate(("Fx", "Fy", "Fz"))
            },
        },
    }
    if normalized_elements:
        mse = normalized_squared_error / normalized_elements
        aggregate["regression"]["normalized"] = {
            "mae": normalized_abs_error / normalized_elements,
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
        }
    curve_arrays: dict[str, np.ndarray] = {"mean_fpr": roc_grid}
    macro_curves: list[np.ndarray] = []
    for name in CLASS_NAMES:
        curves = roc_tprs[name]
        if not curves:
            continue
        mean_tpr = np.mean(curves, axis=0)
        mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
        curve_arrays[f"{name}_mean_tpr"] = mean_tpr
        aggregate["classification"][f"{name}_mean_roc_auc"] = float(
            np.sum((mean_tpr[1:] + mean_tpr[:-1]) * np.diff(roc_grid) * 0.5)
        )
        macro_curves.append(mean_tpr)
    if macro_curves:
        macro_tpr = np.mean(macro_curves, axis=0)
        macro_tpr[0], macro_tpr[-1] = 0.0, 1.0
        curve_arrays["macro_mean_tpr"] = macro_tpr
        aggregate["classification"]["macro_mean_roc_auc"] = float(
            np.sum((macro_tpr[1:] + macro_tpr[:-1]) * np.diff(roc_grid) * 0.5)
        )
    roc_file = (
        "cross-validation-mean-roc.npz"
        if mode == "full"
        else f"cross-validation-mean-roc-{mode}.npz"
    )
    save_arrays(output_dir / roc_file, curve_arrays)
    aggregate["classification"]["mean_roc_curve_file"] = roc_file
    aggregate["evaluation_mode"] = mode
    return aggregate


def _load_finished_fold(
    output_dir: Path,
    test_tool: str,
    expected_config: dict[str, Any],
    expected_settings: dict[str, Any],
    epochs: int,
    batch_size: int,
    expected_spectrogram: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Reuse a fold only when its recorded training and data settings match.

    Photo folds record no ``spectrogram`` entry, so the two image sources never mix.
    """
    path = output_dir / f"fold-{test_tool}.json"
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as file:
            result = json.load(file)
        if not isinstance(result, dict):
            return None
        artifacts = result["artifacts"]
        if not isinstance(artifacts, dict):
            return None
        for artifact in artifacts.values():
            if not isinstance(artifact, str):
                return None
            artifact_path = (output_dir / artifact).resolve()
            if not artifact_path.is_relative_to(output_dir.resolve()) or not artifact_path.is_file():
                return None
        if result["config"] != expected_config:
            return None
        if result.get("spectrogram") != expected_spectrogram:
            return None
        # Folds written before missing-modality evaluation existed must be retrained.
        modality_evaluations = result["modality_evaluations"]
        for mode in EVALUATION_MODES:
            if set(modality_evaluations[mode]) != {"train", "validation", "test"}:
                return None
            if mode != "full" and f"{mode}_predictions" not in artifacts:
                return None
        for dotted_key, expected in expected_settings.items():
            value: Any = result
            for key in dotted_key.split("."):
                value = value[key]
            if value != expected:
                return None
        training = result["training"]
        if training["epochs"] != epochs or training["batch_size"] != batch_size:
            return None
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return result


def run_cross_validation(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    input_len: int = 96,
    horizon: int = 96,
    stride: int | None = None,
    decimation: int = 1,
    label_column: str = "image_label",
    dedup_identical_forces: bool = True,
    eval_dedup_identical_forces: bool | None = None,
    trim_pass: bool = True,
    image_encoder: str = "efficientnet_b0",
    d_model: int = 32,
    prediction_head: str = "direct",
    device_name: str = "cuda",
    seed: int = 42,
    max_train_windows: int | None = None,
    max_eval_windows: int | None = None,
    pool: int = 1,
    image_lag: int = 0,
    select_best: bool = True,
    accumulation_steps: int = 1,
    num_workers: int = 0,
    resume: bool = True,
    config_overrides: dict[str, Any] | None = None,
    spectrogram: SpectrogramSettings | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    spectrogram_record = asdict(spectrogram) if spectrogram is not None else None
    output_path.mkdir(parents=True, exist_ok=True)
    if eval_dedup_identical_forces is None:
        eval_dedup_identical_forces = dedup_identical_forces
    tools = discover_tools(data_dir)
    folds = list(leave_one_tool_out_folds(tools))
    source_signature = nonastreda_source_signature(data_dir)
    results: list[dict[str, Any]] = []
    train_stride = stride if stride is not None else input_len + horizon
    eval_stride = max(train_stride, horizon)
    # The same builder as train_fold, so spectrogram runs (image_channels 7) match too.
    expected_config = asdict(
        model_config(
            d_model=d_model,
            input_len=input_len,
            horizon=horizon,
            image_encoder=image_encoder,
            prediction_head=prediction_head,
            config_overrides=config_overrides,
            spectrogram=spectrogram,
        )
    )
    tracker = ExperimentTracker(
        output_path,
        {
            "dataset": "Nonastreda",
            "data_dir": str(Path(data_dir).resolve()),
            "source_signature": source_signature,
            "protocol": "leave-one-tool-out with cyclic validation tool",
            "tools": tools,
            "folds": [
                {
                    "train_tools": list(fold.train_tools),
                    "validation_tool": fold.validation_tools[0],
                    "test_tool": fold.test_tools[0],
                }
                for fold in folds
            ],
            "training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "stride": train_stride,
                "evaluation_stride": eval_stride,
                "accumulation_steps": accumulation_steps,
                "seed": seed,
                "device": device_name,
            },
            "preprocessing": {
                "decimation": decimation,
                "pool": pool,
                "image_lag": image_lag,
                "label_column": label_column,
                "dedup_identical_forces": dedup_identical_forces,
                "eval_dedup_identical_forces": eval_dedup_identical_forces,
                "trim_pass": trim_pass,
                # Recorded only when used, so photo runs keep their earlier signature.
                **({"spectrogram": spectrogram_record} if spectrogram_record else {}),
            },
            "model": expected_config,
            "ablation": expected_config["ablation"],
            "evaluation_modes": ["full", "sensor_missing", "image_missing"],
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda_runtime": torch.version.cuda,
                "device": device_name,
                "gpu_name": (
                    torch.cuda.get_device_name(torch.device(device_name))
                    if device_name.startswith("cuda") and torch.cuda.is_available()
                    else None
                ),
                "gpu_total_memory_bytes": (
                    torch.cuda.get_device_properties(torch.device(device_name)).total_memory
                    if device_name.startswith("cuda") and torch.cuda.is_available()
                    else None
                ),
            },
            "window_caps": {
                "train": max_train_windows,
                "evaluation": max_eval_windows,
            },
        },
    )

    for fold_index, fold in enumerate(folds):
        test_tool = fold.test_tools[0]
        print(
            f"Fold {fold_index + 1}/{len(folds)}: "
            f"train={len(fold.train_tools)} tools, "
            f"validation={fold.validation_tools[0]}, test={test_tool}",
            flush=True,
        )
        finished = (
            _load_finished_fold(
                output_path,
                test_tool,
                expected_config,
                {
                    "source_signature": source_signature,
                    "train_tools": list(fold.train_tools),
                    "validation_tool": fold.validation_tools[0],
                    "decimation": decimation,
                    "pool": pool,
                    "image_lag": image_lag,
                    "label_column": label_column,
                    "dedup_identical_forces": dedup_identical_forces,
                    "eval_dedup_identical_forces": eval_dedup_identical_forces,
                    "trim_pass": trim_pass,
                    "stride": train_stride,
                    "evaluation_stride": eval_stride,
                    "model_selection": (
                        "best_validation_loss" if select_best else "last_epoch"
                    ),
                    "max_train_windows": max_train_windows,
                    "max_eval_windows": max_eval_windows,
                    "training.learning_rate": learning_rate,
                    "training.accumulation_steps": accumulation_steps,
                    "training.num_workers": num_workers,
                    "training.seed": seed + fold_index,
                    "training.device": device_name,
                },
                epochs,
                batch_size,
                spectrogram_record,
            )
            if resume
            else None
        )
        if finished is not None:
            print(f"  {test_tool}: reusing a completed fold with matching settings", flush=True)
            tracker.log(
                split="test",
                fold=test_tool,
                epoch=finished.get("selected_epoch"),
                metrics=finished["test"],
                context={"reused_completed_fold": True, "evaluation_mode": "full"},
            )
            results.append(finished)
            continue
        result = train_fold(
            data_dir=data_dir,
            test_tool=test_tool,
            output_dir=output_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            input_len=input_len,
            horizon=horizon,
            stride=stride,
            decimation=decimation,
            label_column=label_column,
            dedup_identical_forces=dedup_identical_forces,
            eval_dedup_identical_forces=eval_dedup_identical_forces,
            trim_pass=trim_pass,
            image_encoder=image_encoder,
            d_model=d_model,
            prediction_head=prediction_head,
            max_train_windows=max_train_windows,
            max_eval_windows=max_eval_windows,
            device_name=device_name,
            seed=seed + fold_index,
            pool=pool,
            image_lag=image_lag,
            select_best=select_best,
            accumulation_steps=accumulation_steps,
            num_workers=num_workers,
            resume=resume,
            config_overrides=config_overrides,
            tracker=tracker,
            spectrogram=spectrogram,
        )
        results.append(result)
        partial = {
            "completed_folds": [entry["test_tool"] for entry in results],
            "cv_mean_std": _summarize_folds(results),
            "paper_horizons_mean_std": _summarize_paper_horizons(results),
        }
        with (output_path / "cross-validation-progress.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(partial, file, indent=2)

    summary = {
        "protocol": "10-fold leave-one-tool-out; 8 train tools, next cyclic tool validation, 1 test tool",
        "prediction_head": prediction_head,
        "trim_pass": trim_pass,
        "dedup_identical_forces": dedup_identical_forces,
        "eval_dedup_identical_forces": eval_dedup_identical_forces,
        "pass_trimming": "span" if trim_pass else "none",
        "pool": pool,
        "image_lag": image_lag,
        **({"spectrogram": spectrogram_record} if spectrogram_record else {}),
        "model_selection": (
            "best_validation_loss" if select_best else "last_epoch"
        ),
        "source_signature": source_signature,
        "config": expected_config,
        "ablation": expected_config["ablation"],
        "tools": tools,
        "folds": results,
        "cv_mean_std": _summarize_folds(results),
        "paper_horizons_mean_std": _summarize_paper_horizons(results),
        "modality_evaluations": _summarize_modality_evaluations(results),
        "out_of_fold": _aggregate_out_of_fold(results, output_path),
        "out_of_fold_by_mode": {
            mode: _aggregate_out_of_fold(results, output_path, mode)
            for mode in EVALUATION_MODES
            if mode != "full"
        },
    }
    with (output_path / "cross-validation-results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)
    tracker.log(
        split="cross_validation",
        metrics={
            "cv_mean_std": summary["cv_mean_std"],
            "paper_horizons_mean_std": summary["paper_horizons_mean_std"],
            "modality_evaluations": summary["modality_evaluations"],
            "out_of_fold": summary["out_of_fold"],
            "out_of_fold_by_mode": summary["out_of_fold_by_mode"],
        },
        context={
            "folds": len(results),
            "tools": tools,
            "completed_folds": [entry["test_tool"] for entry in results],
        },
    )
    return summary
