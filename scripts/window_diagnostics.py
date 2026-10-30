"""Diagnostics behind the training-side fixes, from a finished run's saved predictions.

1. Level offset (motivates RevIN): the share of each fold's z-scored forecast MSE that is
   a constant offset per window and channel, mean_t(error)^2, against the share that
   varies within the window. A large offset share means the forecast is at the wrong
   force level for that tool, which instance normalization addresses.
   With ``--data-dir`` it also compares the model's window level with the level RevIN
   would anchor to, the input window's mean (both against the target window's mean).
2. With ``--data-dir`` (rebuilds each fold's test windows; loads the raw data once):
   - the run-number baseline: each target step's class predicted from its run number
     alone, by the training tools' majority class per run (the shortcut the metadata
     code offers);
   - accuracy on target steps whose label differs from the shown photo's, for argmax
     and for the ordinal + monotone rule (``omnifuser.decision``).

Example:
    python scripts/window_diagnostics.py experiment_results/main/full-model --data-dir Nonastreda_Data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from omnifuser.decision import argmax_decode, monotone_ordinal_decode

FOLDS = tuple(f"T{index}" for index in range(1, 11))


def level_offset(run: Path) -> dict[str, Any]:
    per_fold: dict[str, dict[str, float]] = {}
    for fold in FOLDS:
        scale = np.asarray(json.loads((run / f"fold-{fold}.json").read_text(encoding="utf-8"))["normalizer_scale"])
        with np.load(run / f"fold-{fold}-predictions.npz") as arrays:
            error = (arrays["force_predictions"] - arrays["force_targets"]) / scale
        offset = error.mean(axis=1, keepdims=True)
        mse = float(np.square(error).mean())
        per_fold[fold] = {
            "mse": mse,
            "offset_mse": float(np.square(offset).mean()),
            "offset_share": float(np.square(offset).mean() / mse),
            "first_12_leads_mse": float(np.square(error[:, :12]).mean()),
        }
    pooled = {key: float(np.mean([fold[key] for fold in per_fold.values()])) for key in ("mse", "offset_mse")}
    pooled["offset_share"] = pooled["offset_mse"] / pooled["mse"]
    return {"per_fold": per_fold, "fold_mean": pooled}


def window_baselines(run: Path, data_dir: Path) -> dict[str, Any]:
    from omnifuser.config import ModelConfig
    from omnifuser.data import ForceNormalizer
    from omnifuser.evaluation import photo_change_report, run_number_baseline_accuracy
    from omnifuser.train import load_timelines, window_dataset

    first = json.loads((run / "fold-T1.json").read_text(encoding="utf-8"))
    if first.get("spectrogram"):
        raise ValueError("photo runs only")
    timelines, _, _ = load_timelines(
        data_dir, label_column=first["label_column"], decimation=first["decimation"],
        dedup_identical_forces=first["eval_dedup_identical_forces"], trim_pass=first["trim_pass"],
        pool=first["pool"], image_lag=first["image_lag"],
    )
    if first["dedup_identical_forces"] != first["eval_dedup_identical_forces"]:
        train_timelines_all, _, _ = load_timelines(
            data_dir, label_column=first["label_column"], decimation=first["decimation"],
            dedup_identical_forces=first["dedup_identical_forces"], trim_pass=first["trim_pass"],
            pool=first["pool"], image_lag=first["image_lag"],
        )
    else:
        train_timelines_all = timelines
    by_tool = {timeline.tool: timeline for timeline in timelines}
    train_by_tool = {timeline.tool: timeline for timeline in train_timelines_all}
    per_fold: dict[str, dict[str, Any]] = {}
    for fold in FOLDS:
        result = json.loads((run / f"fold-{fold}.json").read_text(encoding="utf-8"))
        config = ModelConfig(**result["config"])
        normalizer = ForceNormalizer(np.asarray(result["normalizer_mean"], dtype=np.float32),
                                     np.asarray(result["normalizer_scale"], dtype=np.float32))
        dataset = window_dataset([by_tool[fold]], normalizer, config, result["evaluation_stride"])
        with np.load(run / f"fold-{fold}-predictions.npz") as arrays:
            probabilities, targets = arrays["class_probabilities"], arrays["class_targets"]
            forecasts = arrays["force_predictions"]
        items = [dataset[index] for index in range(len(dataset))]
        shown = np.asarray([item["shown_label"] for item in items])
        rebuilt = np.stack([item["class_target"] for item in items])
        # The level RevIN anchors to: the input window's mean, against the target's mean.
        input_level = np.stack([item["force"].mean(axis=0) for item in items])
        target_level = np.stack([item["force_target"].mean(axis=0) for item in items])
        model_level = (forecasts.mean(axis=1) - normalizer.mean) / normalizer.scale
        if not np.array_equal(rebuilt, targets):
            raise ValueError(f"{run} {fold}: rebuilt test windows differ from the saved targets")
        per_fold[fold] = {
            "run_number_baseline_accuracy": run_number_baseline_accuracy(
                [train_by_tool[tool] for tool in result["train_tools"]], dataset),
            "persistence_accuracy": float((targets == shown[:, None]).mean()),
            "argmax": photo_change_report(argmax_decode(probabilities), targets, shown),
            "ordinal+monotone": photo_change_report(monotone_ordinal_decode(probabilities), targets, shown),
            "accuracy": float((argmax_decode(probabilities) == targets).mean()),
            "level_error_model": float(np.square(model_level - target_level).mean()),
            "level_error_input_mean": float(np.square(input_level - target_level).mean()),
        }

    def mean(get: Any) -> float | None:
        values = [get(fold) for fold in per_fold.values()]
        values = [value for value in values if value is not None]
        return float(np.mean(values)) if values else None

    summary = {
        "accuracy": mean(lambda f: f["accuracy"]),
        "run_number_baseline_accuracy": mean(lambda f: f["run_number_baseline_accuracy"]),
        "persistence_accuracy": mean(lambda f: f["persistence_accuracy"]),
        "label_changed_share": mean(lambda f: f["argmax"]["label_changed_share"]),
        "argmax_accuracy_label_changed": mean(lambda f: f["argmax"]["accuracy_label_changed"]),
        "argmax_accuracy_label_unchanged": mean(lambda f: f["argmax"]["accuracy_label_unchanged"]),
        "ordinal_monotone_accuracy_label_changed": mean(lambda f: f["ordinal+monotone"]["accuracy_label_changed"]),
        "ordinal_monotone_accuracy_label_unchanged": mean(lambda f: f["ordinal+monotone"]["accuracy_label_unchanged"]),
        "level_error_model": mean(lambda f: f["level_error_model"]),
        "level_error_input_mean": mean(lambda f: f["level_error_input_mean"]),
    }
    accuracy = [fold["accuracy"] for fold in per_fold.values()]
    shortcut = [fold["run_number_baseline_accuracy"] for fold in per_fold.values()]
    summary["fold_correlation_accuracy_vs_run_number_baseline"] = float(np.corrcoef(accuracy, shortcut)[0, 1])
    return {"per_fold": per_fold, "fold_mean": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, help="also rebuild the test windows for the window baselines")
    parser.add_argument("--output", type=Path, default=Path("experiment_results/diagnostics/window-diagnostics.json"))
    args = parser.parse_args()
    results: dict[str, Any] = {}
    for run in args.runs:
        entry: dict[str, Any] = {"level_offset": level_offset(run)}
        if args.data_dir is not None:
            entry["window_baselines"] = window_baselines(run, args.data_dir)
        results[run.as_posix()] = entry
        print(run.as_posix())
        print("  level offset:", json.dumps(entry["level_offset"]["fold_mean"]))
        print("  per fold offset share:", {fold: round(values["offset_share"], 2)
                                           for fold, values in entry["level_offset"]["per_fold"].items()})
        if "window_baselines" in entry:
            print("  window baselines:", json.dumps(entry["window_baselines"]["fold_mean"], indent=1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
