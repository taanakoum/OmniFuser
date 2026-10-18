from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omnifuser.cli import (
    add_dedup_argument,
    add_eval_dedup_argument,
    add_protocol_arguments,
    protocol_kwargs,
)
from omnifuser.config import ModelConfig
from omnifuser.cross_validation import EVALUATION_MODES, run_cross_validation

# Paper Figs. 7-8 axes: (run-name prefix, ModelConfig field, paper symbol, values).
SENSITIVITY_AXES = (
    ("landmarks", "num_landmarks", "k", (2, 4, 8, 16)),
    ("model-width", "d_model", "d", (16, 32, 64, 128)),
    ("gat-heads", "num_gat_heads", "GAT heads", (2, 4, 8, 16)),
    ("rtd-rank", "rtd_rank", "r", (4, 8, 16, 32)),
    ("gat-layers", "num_gat_layers", "GAT layers", (2, 3, 4)),
    ("rtd-stride", "rtd_stride", "s_l", (1, 2, 3)),
    ("mrte-layers", "num_rtd_layers", "L", (1, 2, 3, 4)),
    ("refinement-rounds", "num_fusion_layers", "n", (1, 2, 3)),
)
DEFAULT_MODALITY_DROPOUT_RATES = (0.2,)
SUMMARY_METRICS = (
    "classification_accuracy",
    "classification_precision_macro",
    "classification_recall_macro",
    "classification_f1_macro",
    "classification_roc_auc_macro",
    "regression_mse_normalized",
    "regression_mae_normalized",
    "regression_mse",
    "regression_mae",
)


def _ablation_runs() -> list[dict[str, Any]]:
    return [
        {"name": "full-model", "group": "paper-ablation", "ablation": "none"},
        {"name": "without-rr", "group": "paper-ablation", "ablation": "without_rr"},
        {"name": "without-mrte", "group": "paper-ablation", "ablation": "without_mrte"},
        {
            "name": "without-rr-and-mrte",
            "group": "paper-ablation",
            "ablation": "without_rr_mrte",
        },
        {"name": "concat-c2f", "group": "paper-ablation", "ablation": "concat_c2f"},
        {"name": "without-c2f", "group": "paper-ablation", "ablation": "without_c2f"},
        {"name": "without-als", "group": "paper-ablation", "ablation": "without_als"},
        {
            "name": "without-hybrid-gate",
            "group": "paper-ablation",
            "ablation": "without_hybrid_gate",
        },
        {"name": "moving-average-trend", "group": "paper-ablation", "ablation": "moving_average"},
    ]


def _sensitivity_runs(baseline: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """One baseline plus each off-baseline value; on-baseline points reuse the baseline."""
    reference = baseline or asdict(ModelConfig())
    runs: list[dict[str, Any]] = [
        {"name": "sensitivity-baseline", "group": "parameter-sensitivity", "overrides": {}}
    ]
    for axis_name, field, _, values in SENSITIVITY_AXES:
        for value in values:
            if value == reference[field]:
                continue
            runs.append(
                {
                    "name": f"{axis_name}-{value}",
                    "group": "parameter-sensitivity",
                    "overrides": {field: value},
                }
            )
    return runs


def _missing_modality_runs(
    rates: tuple[float, ...] = DEFAULT_MODALITY_DROPOUT_RATES,
    include_reference: bool = True,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if include_reference:
        runs.append(
            {
                "name": "no-modality-dropout",
                "group": "missing-modality-robustness",
                "overrides": {"modality_dropout": 0.0},
            }
        )
    for rate in rates:
        runs.append(
            {
                "name": f"modality-dropout-{rate:g}".replace(".", "-"),
                "group": "missing-modality-robustness",
                "overrides": {"modality_dropout": rate},
            }
        )
    return runs


def _deduplication_runs(eval_dedup: bool = True) -> list[dict[str, Any]]:
    """Identical runs except whether duplicate recordings are merged for training.

    Both arms are evaluated with the same duplicate handling (deduplicated by
    default), so they are scored on identical validation and test windows.
    """
    return [
        {
            "name": f"dedup-{str(value).lower()}",
            "group": "duplicate-handling",
            "data_overrides": {
                "dedup_identical_forces": value,
                "eval_dedup_identical_forces": eval_dedup,
            },
        }
        for value in (True, False)
    ]


def _experiment_runs(
    study: str,
    baseline: dict[str, Any] | None = None,
    dropout_rates: tuple[float, ...] = DEFAULT_MODALITY_DROPOUT_RATES,
    eval_dedup: bool = True,
) -> list[dict[str, Any]]:
    if study == "ablations":
        return _ablation_runs()
    if study == "sensitivity":
        return _sensitivity_runs(baseline)
    if study == "missing-modality":
        return _missing_modality_runs(dropout_rates)
    if study == "deduplication":
        return _deduplication_runs(eval_dedup)
    # In the combined study the ablation full model is the shared reference arm.
    return (
        _ablation_runs()
        + _sensitivity_runs(baseline)[1:]
        + _missing_modality_runs(dropout_rates, include_reference=False)
    )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _write_suite_summary(study_root: Path, manifest: dict[str, Any]) -> Path:
    """One row per variant with fold mean/std test metrics for every input mode."""
    fieldnames = [
        "group", "variant", "status", "changed_settings", "dedup_identical_forces",
        "eval_dedup_identical_forces", "retained_passes", "evaluation_passes",
        "train_windows.mean", "test_windows.mean",
    ]
    for mode in EVALUATION_MODES:
        for metric in SUMMARY_METRICS:
            fieldnames += [f"{mode}.{metric}.mean", f"{mode}.{metric}.std"]
    rows: list[dict[str, Any]] = []
    for key, record in manifest["runs"].items():
        group, variant = key.split("/", 1)
        row: dict[str, Any] = {
            "group": group,
            "variant": variant,
            "status": record.get("status"),
            "changed_settings": json.dumps(record.get("changed_settings", {}), sort_keys=True),
            "dedup_identical_forces": record.get("dedup_identical_forces"),
            "eval_dedup_identical_forces": record.get("eval_dedup_identical_forces"),
        }
        result_file = record.get("result_file")
        if result_file and (study_root / result_file).is_file():
            with (study_root / result_file).open(encoding="utf-8") as file:
                results = json.load(file)
            evaluations = results.get("modality_evaluations", {})
            folds = results.get("folds", [])
            if folds:
                row["retained_passes"] = folds[0].get("retained_records")
                row["evaluation_passes"] = folds[0].get("evaluation_retained_records")
                for split in ("train", "test"):
                    counts = [fold.get("window_counts", {}).get(split) for fold in folds]
                    counts = [count for count in counts if count is not None]
                    row[f"{split}_windows.mean"] = sum(counts) / len(counts) if counts else None
            for mode in EVALUATION_MODES:
                test = evaluations.get(mode, {}).get("test", {})
                for metric in SUMMARY_METRICS:
                    values = test.get(metric, {})
                    row[f"{mode}.{metric}.mean"] = values.get("mean")
                    row[f"{mode}.{metric}.std"] = values.get("std")
        rows.append(row)
    path = study_root / "suite-summary.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run paper-aligned ablation, sensitivity, or missing-modality studies "
            "as isolated 10-fold experiment directories."
        )
    )
    parser.add_argument(
        "--study",
        choices=("ablations", "sensitivity", "missing-modality", "deduplication", "all"),
        default="ablations",
        help="'all' runs the three paper studies; 'deduplication' compares --dedup-identical-forces True and False",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("Nonastreda_Data"))
    parser.add_argument("--output-root", type=Path, default=Path("experiment_results"))
    parser.add_argument(
        "--run-id",
        help="reuse this named suite directory after interruption; settings must match",
    )
    parser.add_argument(
        "--modality-dropout-rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_MODALITY_DROPOUT_RATES),
        help="training modality-dropout rates for the missing-modality study",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument(
        "--trim-pass", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--label-column",
        choices=("image_label", "tool_label"),
        default="image_label",
    )
    add_dedup_argument(parser)
    add_eval_dedup_argument(parser)
    parser.add_argument(
        "--image-encoder",
        choices=("efficientnet_b0", "compact"),
        default="efficientnet_b0",
    )
    parser.add_argument(
        "--prediction-head",
        choices=("paper_literal", "lead_conditioned", "direct"),
        default="direct",
    )
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--max-eval-windows", type=int)
    add_protocol_arguments(parser)
    # The suite's other defaults (30 epochs, stride 24, horizon 96) assume 1 s steps;
    # the photo is the same pass's, as in the paper (lag 1 is an experimental run).
    parser.set_defaults(pool=1000, image_lag=0)
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or timestamp
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run-id must be one directory name, not a path")
    if any(not 0.0 < rate < 1.0 for rate in args.modality_dropout_rates):
        raise ValueError("modality dropout rates must be in (0, 1)")
    study_root = args.output_root / args.study / run_id
    manifest_path = study_root / "suite.json"
    common_protocol = protocol_kwargs(args)
    baseline = asdict(
        ModelConfig(
            d_model=args.d_model,
            input_len=args.input_len,
            horizon=args.horizon,
            image_encoder=args.image_encoder,
            prediction_head=args.prediction_head,
            **common_protocol["config_overrides"],
        )
    )
    runs = _experiment_runs(
        args.study,
        baseline,
        tuple(args.modality_dropout_rates),
        eval_dedup=(
            True if args.eval_dedup_identical_forces is None else args.eval_dedup_identical_forces
        ),
    )
    shared_settings = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    shared_settings["run_id"] = None
    if study_root.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"experiment output directory already exists: {study_root}")
        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        if (
            manifest.get("study") != args.study
            or manifest.get("shared_settings") != shared_settings
        ):
            raise ValueError(
                f"cannot resume {study_root}; study settings differ from suite.json"
            )
    else:
        study_root.mkdir(parents=True)
        manifest = {
            "format_version": 2,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "study": args.study,
            "paper_scope": "OmniFuser ablation, sensitivity, and missing-modality studies on Nonastreda",
            "interpretation_notes": [
                "Sensitivity symbols map to ModelConfig fields: "
                + ", ".join(f"{symbol} -> {field}" for _, field, symbol, _ in SENSITIVITY_AXES)
                + ".",
                (
                    "Sensitivity values equal to the baseline are not retrained; read them from "
                    "sensitivity-baseline (or paper-ablation/full-model in the combined study)."
                ),
                (
                    "Missing-modality rates are explicit choices because the paper does not report one; "
                    "every variant is also evaluated with sensor-only and image-only masking at inference."
                ),
                (
                    "w/o-ALS uses fixed segment-mean landmarks, w/o-C2F an equal-weight stream mean, "
                    "w/o-HG a fixed 0.5 mixture, and re-LR-MA a three-token moving average; the paper "
                    "does not specify these operators."
                ),
            ],
            "baseline_config": baseline,
            "shared_settings": shared_settings,
            "runs": {},
        }
    manifest["status"] = "running"
    _write_manifest(manifest_path, manifest)
    for spec in runs:
        group = str(spec["group"])
        name = str(spec["name"])
        run_path = study_root / group / name
        run_key = f"{group}/{name}"
        existing_record = manifest["runs"].get(run_key)
        if existing_record is not None and existing_record.get("status") == "completed":
            continue
        run_path.mkdir(parents=True, exist_ok=True)
        config_overrides = dict(common_protocol["config_overrides"])
        changed_settings = dict(spec.get("overrides", {}))
        if "ablation" in spec:
            changed_settings["ablation"] = spec["ablation"]
        config_overrides.update(changed_settings)
        model_width = int(config_overrides.pop("d_model", args.d_model))
        # Data settings vary per variant only in the deduplication study; they are
        # passed to the loader, never to ModelConfig.
        data_overrides = dict(spec.get("data_overrides", {}))
        dedup_identical_forces = bool(
            data_overrides.get("dedup_identical_forces", args.dedup_identical_forces)
        )
        eval_dedup_identical_forces = data_overrides.get(
            "eval_dedup_identical_forces", args.eval_dedup_identical_forces
        )
        if eval_dedup_identical_forces is None:
            eval_dedup_identical_forces = dedup_identical_forces
        changed_settings.update(data_overrides)
        run_record: dict[str, Any] = {
            "status": "running",
            "group": group,
            "changed_settings": changed_settings,
            "config_overrides": config_overrides,
            "d_model": model_width,
            "dedup_identical_forces": dedup_identical_forces,
            "eval_dedup_identical_forces": eval_dedup_identical_forces,
            "output_dir": str(run_path.relative_to(study_root)),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest["runs"][run_key] = run_record
        _write_manifest(manifest_path, manifest)
        print(f"=== {run_key} ({changed_settings or 'baseline'}) ===", flush=True)
        run_protocol = common_protocol.copy()
        run_protocol["config_overrides"] = config_overrides
        try:
            summary = run_cross_validation(
                data_dir=args.data_dir,
                output_dir=run_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                input_len=args.input_len,
                horizon=args.horizon,
                stride=args.stride,
                decimation=args.decimation,
                label_column=args.label_column,
                dedup_identical_forces=dedup_identical_forces,
                eval_dedup_identical_forces=eval_dedup_identical_forces,
                trim_pass=args.trim_pass,
                image_encoder=args.image_encoder,
                d_model=model_width,
                prediction_head=args.prediction_head,
                device_name=args.device,
                seed=args.seed,
                max_train_windows=args.max_train_windows,
                max_eval_windows=args.max_eval_windows,
                **run_protocol,
            )
        except BaseException as exception:
            run_record["status"] = "failed"
            run_record["error"] = f"{type(exception).__name__}: {exception}"
            _write_manifest(manifest_path, manifest)
            _write_suite_summary(study_root, manifest)
            raise
        run_record["status"] = "completed"
        run_record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_record["result_file"] = str(
            (run_path / "cross-validation-results.json").relative_to(study_root)
        )
        run_record["test_metrics"] = summary["cv_mean_std"]
        _write_manifest(manifest_path, manifest)
        _write_suite_summary(study_root, manifest)
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest_path, manifest)
    summary_path = _write_suite_summary(study_root, manifest)
    print(
        json.dumps(
            {"study_dir": str(study_root), "runs": len(runs), "summary": str(summary_path)},
            indent=2,
        )
    )
    return study_root


if __name__ == "__main__":
    main()
