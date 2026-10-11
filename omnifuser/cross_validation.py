from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from omnifuser.data import CLASS_NAMES, leave_one_tool_out_folds
from omnifuser.train import train_fold


RECORD_IMAGE = re.compile(r"^T(?P<tool>\d+)R\d+B\d+$")


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
        "classification_roc_auc_macro": ("test", "classification", "roc_auc_ovr", "macro"),
        "force_mae": ("test", "regression", "overall", "mae"),
        "force_mse": ("test", "regression", "overall", "mse"),
        "force_rmse": ("test", "regression", "overall", "rmse"),
    }
    summary: dict[str, Any] = {"folds": len(results), "metrics": {}}
    for name, path in scalar_paths.items():
        per_fold: list[float | None] = []
        for result in results:
            value: Any = result
            for key in path:
                value = value[key]
            per_fold.append(None if value is None else float(value))
        values = [value for value in per_fold if value is not None]
        summary["metrics"][name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "per_fold": per_fold,
        }
    return summary


def _aggregate_out_of_fold(
    results: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    regression_abs_error = 0.0
    regression_squared_error = 0.0
    regression_elements = 0
    channel_abs_error = np.zeros(3, dtype=np.float64)
    channel_squared_error = np.zeros(3, dtype=np.float64)
    channel_elements = 0
    roc_grid = np.linspace(0.0, 1.0, 1001)
    roc_tprs: dict[str, list[np.ndarray]] = {name: [] for name in CLASS_NAMES}
    roc_fold_auc: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}

    for result in results:
        tool = result["test_tool"]
        confusion += np.asarray(
            result["test"]["classification"]["confusion_matrix"], dtype=np.int64
        )
        prediction_path = output_dir / result["artifacts"]["test_predictions"]
        with np.load(prediction_path) as predictions:
            error = predictions["force_predictions"] - predictions["force_targets"]
            regression_abs_error += float(np.abs(error).sum())
            regression_squared_error += float(np.square(error).sum())
            regression_elements += error.size
            channel_abs_error += np.abs(error).sum(axis=(0, 1))
            channel_squared_error += np.square(error).sum(axis=(0, 1))
            channel_elements += error.shape[0] * error.shape[1]

        roc_path = output_dir / result["artifacts"]["roc_curves"]
        with np.load(roc_path) as curves:
            for name in CLASS_NAMES:
                auc = result["test"]["classification"]["roc_auc_ovr"]["per_class"][name]
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
    np.savez_compressed(output_dir / "cross-validation-mean-roc.npz", **curve_arrays)
    aggregate["classification"]["mean_roc_curve_file"] = (
        "cross-validation-mean-roc.npz"
    )
    return aggregate


def run_cross_validation(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    input_len: int = 96,
    horizon: int = 24,
    stride: int | None = None,
    decimation: int = 1,
    label_column: str = "image_label",
    dedup_identical_forces: bool = True,
    trim_pass: bool = True,
    image_encoder: str = "efficientnet_b0",
    d_model: int = 32,
    prediction_head: str = "lead_conditioned",
    device_name: str = "cuda",
    seed: int = 42,
    max_train_windows: int | None = None,
    max_eval_windows: int | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tools = discover_tools(data_dir)
    folds = list(leave_one_tool_out_folds(tools))
    results: list[dict[str, Any]] = []

    for fold_index, fold in enumerate(folds):
        test_tool = fold.test_tools[0]
        print(
            f"Fold {fold_index + 1}/{len(folds)}: "
            f"train={len(fold.train_tools)} tools, "
            f"validation={fold.validation_tools[0]}, test={test_tool}",
            flush=True,
        )
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
            trim_pass=trim_pass,
            image_encoder=image_encoder,
            d_model=d_model,
            prediction_head=prediction_head,
            max_train_windows=max_train_windows,
            max_eval_windows=max_eval_windows,
            device_name=device_name,
            seed=seed + fold_index,
        )
        results.append(result)
        partial = {
            "completed_folds": [entry["test_tool"] for entry in results],
            "cv_mean_std": _summarize_folds(results),
        }
        with (output_path / "cross-validation-progress.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(partial, file, indent=2)

    summary = {
        "protocol": "10-fold leave-one-tool-out; 8 train tools, next cyclic tool validation, 1 test tool",
        "prediction_head": prediction_head,
        "trim_pass": trim_pass,
        "pass_trimming": "span" if trim_pass else "none",
        "tools": tools,
        "folds": results,
        "cv_mean_std": _summarize_folds(results),
        "out_of_fold": _aggregate_out_of_fold(results, output_path),
    }
    with (output_path / "cross-validation-results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)
    return summary
