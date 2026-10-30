"""Analyse the "fixes" study over its seeds and apply the pre-declared decision rules.

Layout (TRAINING_RUNBOOK_RUCHE_FIXES.md): one suite per seed,
``<root>/<seed run-id>/fixes/<arm>/`` with 10 folds each. For every arm and fold the
metrics are averaged over the seeds; each arm is then compared with ``reference`` fold
by fold (paired Wilcoxon over the 10 folds). The class probabilities of the seeds are
also averaged (a seed ensemble) and scored.

Decision rules (fixed before stage 1 runs; see the runbook, §5):

* accuracy arms: adopt when seed-averaged accuracy gains >= 0.01 with p < 0.10;
* ``revin``: adopt when normalized force MSE falls by >= 3 % with p < 0.10 and accuracy
  does not fall by more than 0.01;
* ``metadata-*``: adopt when accuracy does not fall by more than 0.01 and the
  training-test accuracy gap shrinks; "none" is preferred over "position" when both pass
  (fewer codes, no shortcut);
* ``coral``: adopt when accuracy does not fall by more than 0.01 and "used" recall gains
  >= 0.05 with p < 0.10, or accuracy gains as for the accuracy arms.

It prints the flags of the adopted arms: the stage-2 "combined" run.

Example:
    python scripts/fixes_analysis.py --root experiment_results/fixes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from omnifuser.decision import argmax_decode, decision_scores, monotone_ordinal_decode

FOLDS = tuple(f"T{index}" for index in range(1, 11))
FLAGS = {
    "reference": "",
    "modality-dropout-0-2": "--modality-dropout 0.2",
    "metadata-position": "--metadata position",
    "metadata-none": "--metadata none",
    "revin": "--revin",
    "coral": "--class-loss coral",
    "photo-edge-crop": "--image-crop 0.30 0 0.85 1 --image-size 128 640",
    "no-image-gat": "--no-image-gat",
    "finetuned-encoder": "--no-freeze-image-encoder",
}
ACCURACY_GAIN = 0.01
NON_INFERIORITY = 0.01
MSE_RELATIVE_DROP = 0.03
USED_RECALL_GAIN = 0.05
P_LIMIT = 0.10


def _get(data: dict[str, Any], *path: str) -> float | None:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return None if value is None else float(value)


def fold_metrics(run: Path, fold: str) -> tuple[dict[str, float | None], np.ndarray, np.ndarray]:
    result = json.loads((run / f"fold-{fold}.json").read_text(encoding="utf-8"))
    with np.load(run / f"fold-{fold}-predictions.npz") as arrays:
        probabilities, targets = arrays["class_probabilities"], arrays["class_targets"]
    scores = decision_scores(argmax_decode(probabilities), targets)
    train = _get(result, "modality_evaluations", "full", "train", "classification_accuracy")
    metrics: dict[str, float | None] = {
        "accuracy": scores["accuracy"],
        "accuracy_ordinal_monotone": decision_scores(monotone_ordinal_decode(probabilities), targets)["accuracy"],
        "f1_macro": scores["f1_macro"],
        "recall_used": scores["recall_class_1"],
        "mace": scores["mean_absolute_class_error"],
        "force_mse_normalized": _get(result, "test", "regression", "normalized", "mse"),
        "accuracy_label_changed": _get(result, "test", "photo_change", "accuracy_label_changed"),
        "accuracy_photo_masked": _get(result, "modality_evaluations", "image_missing", "test", "classification_accuracy"),
        "train_test_gap": None if train is None else train - scores["accuracy"],
        "run_number_baseline": _get(result, "test", "baselines", "run_number_class_accuracy"),
    }
    return metrics, probabilities, targets


def arm_summary(runs: list[Path]) -> dict[str, Any]:
    """Seed-averaged per-fold metrics and the seed ensemble's per-fold scores."""
    per_fold: dict[str, list[float | None]] = {}
    seed_sd: dict[str, list[float]] = {}
    ensemble: dict[str, list[float]] = {"accuracy": [], "accuracy_ordinal_monotone": [], "f1_macro": []}
    for fold in FOLDS:
        loaded = [fold_metrics(run, fold) for run in runs]
        targets = loaded[0][2]
        for metrics, _, other_targets in loaded:
            if not np.array_equal(other_targets, targets):
                raise ValueError(f"{fold}: seeds have different test windows")
        for key in loaded[0][0]:
            values: list[float] = [value for metrics, _, _ in loaded if (value := metrics[key]) is not None]
            per_fold.setdefault(key, []).append(float(np.mean(values)) if values else None)
            if len(values) > 1:
                seed_sd.setdefault(key, []).append(float(np.std(values, ddof=1)))
        averaged = np.mean([probabilities for _, probabilities, _ in loaded], axis=0)
        scores = decision_scores(argmax_decode(averaged), targets)
        ensemble["accuracy"].append(scores["accuracy"])
        ensemble["f1_macro"].append(scores["f1_macro"])
        ensemble["accuracy_ordinal_monotone"].append(
            decision_scores(monotone_ordinal_decode(averaged), targets)["accuracy"])
    return {
        "seeds": [run.as_posix() for run in runs],
        "per_fold": per_fold,
        "mean": {key: _mean(values) for key, values in per_fold.items()},
        "seed_sd_per_fold_mean": {key: float(np.mean(values)) for key, values in seed_sd.items()},
        "seed_ensemble": {key: float(np.mean(values)) for key, values in ensemble.items()},
        "seed_ensemble_per_fold": ensemble,
    }


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def paired(arm: list[float | None], reference: list[float | None]) -> dict[str, Any] | None:
    pairs = [(a, r) for a, r in zip(arm, reference) if a is not None and r is not None]
    if len(pairs) < 2:
        return None
    diff = np.asarray([a - r for a, r in pairs])
    p_value = None
    if np.any(np.abs(diff) > 1e-12):
        from scipy.stats import wilcoxon
        # SciPy's type stubs do not declare the result's fields.
        p_value = float(wilcoxon(diff).pvalue)  # pyright: ignore[reportAttributeAccessIssue]
    return {"difference": float(diff.mean()), "folds_up": int((diff > 1e-12).sum()),
            "folds_down": int((diff < -1e-12).sum()), "wilcoxon_p": p_value}


def _significant(test: dict[str, Any] | None) -> bool:
    return test is not None and test["wilcoxon_p"] is not None and test["wilcoxon_p"] < P_LIMIT


def decide(name: str, tests: dict[str, dict[str, Any] | None], reference_mse: float | None) -> tuple[bool, str]:
    accuracy = tests["accuracy"]
    if accuracy is None:
        return False, "no accuracy comparison"
    non_inferior = accuracy["difference"] >= -NON_INFERIORITY
    accuracy_gain = accuracy["difference"] >= ACCURACY_GAIN and _significant(accuracy)
    if name == "revin":
        mse = tests["force_mse_normalized"]
        relative = None if mse is None or not reference_mse else -mse["difference"] / reference_mse
        adopted = relative is not None and relative >= MSE_RELATIVE_DROP and _significant(mse) and non_inferior
        return adopted, f"MSE change {-(relative or 0):+.1%}, accuracy {accuracy['difference']:+.3f}"
    if name.startswith("metadata-"):
        gap = tests["train_test_gap"]
        shrinks = gap is not None and gap["difference"] < 0
        return non_inferior and shrinks, (
            f"accuracy {accuracy['difference']:+.3f}, gap change "
            f"{'n/a' if gap is None else format(gap['difference'], '+.3f')}")
    if name == "coral":
        used = tests["recall_used"]
        used_gain = used is not None and used["difference"] >= USED_RECALL_GAIN and _significant(used)
        return (non_inferior and used_gain) or accuracy_gain, (
            f"accuracy {accuracy['difference']:+.3f}, used recall "
            f"{'n/a' if used is None else format(used['difference'], '+.3f')}")
    return accuracy_gain, f"accuracy {accuracy['difference']:+.3f} (p={accuracy['wilcoxon_p']})"


def analyse(root: Path) -> dict[str, Any]:
    seeds = sorted(path for path in root.iterdir() if (path / "fixes" / "reference").is_dir())
    if not seeds:
        raise FileNotFoundError(f"no <seed>/fixes/reference folders under {root}")
    arms = sorted({arm.name for seed in seeds for arm in (seed / "fixes").iterdir()
                   if (arm / "cross-validation-results.json").is_file()})
    summaries = {arm: arm_summary([seed / "fixes" / arm for seed in seeds
                                   if (seed / "fixes" / arm / "cross-validation-results.json").is_file()])
                 for arm in arms}
    reference = summaries["reference"]
    decisions: dict[str, Any] = {}
    for arm, summary in summaries.items():
        if arm == "reference":
            continue
        tests = {key: paired(summary["per_fold"][key], reference["per_fold"][key]) for key in summary["per_fold"]}
        adopted, reason = decide(arm, tests, reference["mean"]["force_mse_normalized"])
        decisions[arm] = {"adopted": adopted, "reason": reason, "tests": tests, "seeds": len(summary["seeds"])}
    if decisions.get("metadata-none", {}).get("adopted") and decisions.get("metadata-position", {}).get("adopted"):
        decisions["metadata-position"]["adopted"] = False
        decisions["metadata-position"]["reason"] += "; superseded by metadata-none"
    flags = " ".join(FLAGS[arm] for arm, decision in decisions.items() if decision["adopted"] and FLAGS.get(arm))
    return {"seeds": [seed.name for seed in seeds], "arms": summaries, "decisions": decisions,
            "stage2_flags": flags}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("experiment_results/fixes"))
    parser.add_argument("--output", type=Path, help="default: <root>/fixes-analysis.json")
    args = parser.parse_args()
    result = analyse(args.root)
    output = args.output or args.root / "fixes-analysis.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"seeds: {', '.join(result['seeds'])}")
    header = f"{'arm':22s} {'acc':>6s} {'ord+mon':>7s} {'used':>6s} {'MSE':>6s} {'gap':>6s} {'changed':>7s} {'ens.acc':>7s}"
    print(header)
    for arm, summary in result["arms"].items():
        mean = summary["mean"]

        def show(key: str, mean: dict[str, Any] = mean) -> str:
            return "   n/a" if mean.get(key) is None else f"{mean[key]:6.3f}"
        print(f"{arm:22s} {show('accuracy')} {show('accuracy_ordinal_monotone'):>7s} {show('recall_used')} "
              f"{show('force_mse_normalized')} {show('train_test_gap')} {show('accuracy_label_changed'):>7s} "
              f"{summary['seed_ensemble']['accuracy']:7.3f}")
    print()
    for arm, decision in result["decisions"].items():
        print(f"{arm:22s} {'ADOPT ' if decision['adopted'] else 'reject'}  {decision['reason']}")
    print(f"\nstage-2 flags: {result['stage2_flags'] or '(none adopted)'}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
