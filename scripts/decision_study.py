"""Evaluate wear-class decision rules on finished runs, without retraining.

Parameter-free rules (``omnifuser.decision``) are applied to every fold's saved test
predictions. With ``--tune-offsets``, per-class log-probability offsets are fitted on
each fold's **validation tool** (predicted with the fold's saved model, which needs the
``fold-Tn.pt`` files and the data) and then applied to that fold's test predictions.

Examples:
    python scripts/decision_study.py                      # every finished run
    python scripts/decision_study.py experiment_results/main/full-model --tune-offsets \
        --data-dir Nonastreda_Data
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from omnifuser.decision import (
    apply_offsets,
    argmax_decode,
    decision_scores,
    fit_class_offsets,
    monotone_decode,
    monotone_ordinal_decode,
    ordinal_median,
)
from omnifuser.evaluation import PAPER_HORIZONS

DECODERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "argmax": argmax_decode,
    "monotone": monotone_decode,
    "ordinal": ordinal_median,
    "ordinal+monotone": monotone_ordinal_decode,
}
REPORTED = ("accuracy", "accuracy_paper_horizons", "precision_macro", "f1_macro",
            "recall_class_1", "mean_absolute_class_error", "changing_windows", "decreasing_windows")


def _fold_paths(run: Path) -> list[Path]:
    return sorted(run.glob("fold-T*.json"), key=lambda p: int(p.stem.split("-T")[1]))


def _scores(predicted: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    scores = decision_scores(predicted, targets)
    horizons = [h for h in PAPER_HORIZONS if h <= targets.shape[1]]
    scores["accuracy_paper_horizons"] = float(np.mean(
        [(predicted[:, :h] == targets[:, :h]).mean() for h in horizons]))
    return scores


def _validation_predictions(run: Path, fold: dict[str, Any], data_dir: Path, cache: dict[str, Any]) -> dict[str, np.ndarray]:
    """Class probabilities and targets on the fold's validation tool, from its saved model."""
    import torch
    from torch.utils.data import DataLoader

    from omnifuser.data import collate_windows
    from omnifuser.model_bundle import load_model
    from omnifuser.train import (
        _run_epoch,
        image_bank_for,
        load_timelines,
        window_dataset,
    )

    tool = fold["test_tool"]
    output = run / "decision" / f"fold-{tool}-validation-predictions.npz"
    if output.is_file():
        with np.load(output) as arrays:
            return {key: arrays[key] for key in arrays.files}
    loaded = load_model(run / f"fold-{tool}.pt")
    data = loaded.data
    key = json.dumps({k: data[k] for k in ("label_column", "decimation", "eval_dedup_identical_forces",
                                           "trim_pass", "pool", "image_lag")}, sort_keys=True)
    if loaded.spectrogram is not None:
        raise ValueError("offset tuning supports photo runs only")
    if key not in cache:  # the raw data is loaded once per data setting, not per fold
        timelines, _, _ = load_timelines(
            data_dir, label_column=data["label_column"], decimation=data["decimation"],
            dedup_identical_forces=data["eval_dedup_identical_forces"], trim_pass=data["trim_pass"],
            pool=data["pool"], image_lag=data["image_lag"],
        )
        cache[key] = {timeline.tool: timeline for timeline in timelines}
    timeline = cache[key][fold["validation_tool"]]
    dataset = window_dataset([timeline], loaded.normalizer, loaded.config, data["evaluation_stride"])
    device = torch.device("cpu")
    bank = image_bank_for([timeline], loaded.config, loaded.model, device)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=partial(
        collate_windows, height=loaded.config.image_height, width=loaded.config.image_width, bank=bank))
    _, predictions = _run_epoch(loaded.model, loader, None, device, loaded.normalizer, collect_predictions=True)
    assert predictions is not None
    arrays = {"class_probabilities": predictions["class_probabilities"],
              "class_targets": predictions["class_targets"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return arrays


def study_run(run: Path, tune_offsets: bool, data_dir: Path | None, cache: dict[str, Any]) -> dict[str, Any]:
    per_fold: dict[str, list[dict[str, float]]] = {}
    offsets: dict[str, list[float]] = {}
    for path in _fold_paths(run):
        fold = json.loads(path.read_text(encoding="utf-8"))
        with np.load(path.with_name(f"{path.stem}-predictions.npz")) as arrays:
            probabilities, targets = arrays["class_probabilities"], arrays["class_targets"]
        variants = {name: decode(probabilities) for name, decode in DECODERS.items()}
        if tune_offsets:
            if data_dir is None:
                raise ValueError("--tune-offsets needs --data-dir")
            validation = _validation_predictions(run, fold, data_dir, cache)
            for name in ("argmax", "monotone"):
                fitted = fit_class_offsets(validation["class_probabilities"], validation["class_targets"],
                                           DECODERS[name])
                offsets.setdefault(name, []).append(fitted.tolist())
                variants[f"offsets+{name}"] = DECODERS[name](apply_offsets(probabilities, fitted))
        for name, predicted in variants.items():
            per_fold.setdefault(name, []).append(_scores(predicted, targets))
    summary: dict[str, Any] = {"folds": len(next(iter(per_fold.values()))), "decoders": {}}
    base = [fold["accuracy"] for fold in per_fold["argmax"]]
    for name, folds in per_fold.items():
        entry = {key: float(np.mean([fold[key] for fold in folds])) for key in REPORTED}
        entry["accuracy_per_fold"] = [fold["accuracy"] for fold in folds]
        if name != "argmax":
            diff = np.asarray(entry["accuracy_per_fold"]) - np.asarray(base)
            entry["folds_better"] = int((diff > 1e-12).sum())
            entry["folds_worse"] = int((diff < -1e-12).sum())
            entry["wilcoxon_p"] = _wilcoxon(diff)
        summary["decoders"][name] = entry
    if offsets:
        summary["fitted_offsets"] = offsets
    return summary


def _wilcoxon(diff: np.ndarray) -> float | None:
    if not np.any(np.abs(diff) > 1e-12):
        return None
    from scipy.stats import wilcoxon
    return float(wilcoxon(diff).pvalue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="run folders (default: every finished run)")
    parser.add_argument("--root", type=Path, default=Path("experiment_results"))
    parser.add_argument("--tune-offsets", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiment_results/decision/decision-study.json"))
    args = parser.parse_args()
    runs = args.runs or sorted(
        path.parent for path in args.root.rglob("cross-validation-results.json")
        if not ({"validation", "pilot", "end_to_end"} & set(path.relative_to(args.root).parts))
    )
    cache: dict[str, Any] = {}
    results = {}
    for run in runs:
        results[run.as_posix()] = study_run(run, args.tune_offsets, args.data_dir, cache)
        main_scores = results[run.as_posix()]["decoders"]
        print(run.as_posix(), " ".join(
            f"{name}={entry['accuracy']:.3f}" for name, entry in main_scores.items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
