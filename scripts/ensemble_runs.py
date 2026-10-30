"""Average the predictions of finished runs that share their test windows (an ensemble).

For every fold, the class probabilities and force forecasts of the given runs are
averaged, then scored like a single run. The members are scored the same way, so the
output also measures how much identical runs differ (replicates) and what averaging them
gains. Runs must have identical test windows: the script checks the class targets.

Examples:
    # replicates: runs with the same settings (and seed), differing only by GPU nondeterminism
    python scripts/ensemble_runs.py experiment_results/main/full-model \
        experiment_results/heads/direct --name replicates
    # seeds of one configuration (TRAINING_RUNBOOK_RUCHE_FIXES.md)
    python scripts/ensemble_runs.py experiment_results/fixes/s*/fixes/reference --name reference
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from omnifuser.decision import argmax_decode, decision_scores, monotone_ordinal_decode

FOLDS = tuple(f"T{index}" for index in range(1, 11))
REPORTED = ("accuracy", "accuracy_ordinal_monotone", "f1_macro", "precision_macro",
            "recall_class_1", "force_mse_normalized")


def _fold_scores(probabilities: np.ndarray, targets: np.ndarray,
                 forecast: np.ndarray, force_targets: np.ndarray, scale: np.ndarray) -> dict[str, float]:
    scores = decision_scores(argmax_decode(probabilities), targets)
    scores["accuracy_ordinal_monotone"] = decision_scores(monotone_ordinal_decode(probabilities), targets)["accuracy"]
    scores["force_mse_normalized"] = float(np.square((forecast - force_targets) / scale).mean())
    return scores


def _load(run: Path, fold: str) -> dict[str, np.ndarray]:
    with np.load(run / f"fold-{fold}-predictions.npz") as arrays:
        return {key: arrays[key] for key in arrays.files}


def ensemble(runs: list[Path]) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("an ensemble needs at least two runs")
    members: dict[str, list[dict[str, float]]] = {run.as_posix(): [] for run in runs}
    combined: list[dict[str, float]] = []
    for fold in FOLDS:
        loaded = [_load(run, fold) for run in runs]
        targets = loaded[0]["class_targets"]
        for run, arrays in zip(runs, loaded):
            if arrays["class_targets"].shape != targets.shape or not np.array_equal(arrays["class_targets"], targets):
                raise ValueError(f"{run} fold {fold} has other test windows than {runs[0]}")
        scale = np.asarray(json.loads((runs[0] / f"fold-{fold}.json").read_text(encoding="utf-8"))["normalizer_scale"])
        force_targets = loaded[0]["force_targets"]
        for run, arrays in zip(runs, loaded):
            members[run.as_posix()].append(_fold_scores(
                arrays["class_probabilities"], targets, arrays["force_predictions"], force_targets, scale))
        combined.append(_fold_scores(
            np.mean([arrays["class_probabilities"] for arrays in loaded], axis=0), targets,
            np.mean([arrays["force_predictions"] for arrays in loaded], axis=0), force_targets, scale))

    def mean(folds: list[dict[str, float]]) -> dict[str, float]:
        return {key: float(np.mean([fold[key] for fold in folds])) for key in REPORTED}

    member_means = {run: mean(folds) for run, folds in members.items()}
    member_fold_mean = {key: np.mean([[fold[key] for fold in folds] for folds in members.values()], axis=0)
                        for key in REPORTED}
    result: dict[str, Any] = {
        "runs": [run.as_posix() for run in runs],
        "ensemble": mean(combined),
        "members": member_means,
        "member_spread": {
            key: {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                  "min": float(np.min(values)), "max": float(np.max(values))}
            for key in REPORTED
            for values in [[scores[key] for scores in member_means.values()]]
        },
        "ensemble_per_fold": {key: [fold[key] for fold in combined] for key in REPORTED},
    }
    comparison: dict[str, Any] = {}
    for key in REPORTED:
        diff = np.asarray(result["ensemble_per_fold"][key]) - member_fold_mean[key]
        comparison[key] = {
            "gain_over_member_mean": float(diff.mean()),
            "folds_better": int((diff > 1e-12).sum()),
            "folds_worse": int((diff < -1e-12).sum()),
            "wilcoxon_p": _wilcoxon(diff),
        }
    result["against_member_mean"] = comparison
    return result


def _wilcoxon(diff: np.ndarray) -> float | None:
    if not np.any(np.abs(diff) > 1e-12):
        return None
    from scipy.stats import wilcoxon
    # SciPy's type stubs do not declare the result's fields.
    return float(wilcoxon(diff).pvalue)  # pyright: ignore[reportAttributeAccessIssue]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run folders with fold-Tn-predictions.npz")
    parser.add_argument("--name", required=True, help="name of the ensemble (output file stem)")
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_results/ensembles"))
    args = parser.parse_args()
    result = ensemble(args.runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.name}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    spread = result["member_spread"]
    for key in REPORTED:
        gain = result["against_member_mean"][key]
        print(f"{key:28s} members {spread[key]['mean']:.4f} ± {spread[key]['std']:.4f} "
              f"[{spread[key]['min']:.4f}, {spread[key]['max']:.4f}]  ensemble {result['ensemble'][key]:.4f}  "
              f"gain {gain['gain_over_member_mean']:+.4f} ({gain['folds_better']}/{gain['folds_worse']} folds, "
              f"p={gain['wilcoxon_p']})")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
