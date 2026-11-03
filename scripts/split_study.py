"""How much the fold split explains: leave-one-tool-out against two easier splits.

Layout (TRAINING_RUNBOOK_RUCHE_FIXES.md): one ``--study splits`` suite per seed,
``<root>/<seed run-id>/splits/<arm>/``. The arms test on different windows, so they are
compared by their means over the seeds (each seed's difference to leave-one-tool-out,
then mean ± SD), not fold by fold. Accuracy, precision, MSE and MAE are means over the
paper's 8 horizons, like the paper's Tables IV-V.

Example:
    python scripts/split_study.py --root experiment_results/splits
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from omnifuser.evaluation import PAPER_HORIZONS, PAPER_OMNIFUSER_RESULTS

ARMS = ("leave-one-tool-out", "within-tool-blocks", "random-windows")
METRICS = ("accuracy", "precision_macro", "mse", "mae", "train_test_gap", "persistence",
           "accuracy_label_changed", "train_windows", "test_windows")


def _horizon_mean(test: dict[str, Any], key: str) -> float:
    return float(np.mean([test["paper_horizons"][str(h)][key] for h in PAPER_HORIZONS]))


def run_metrics(run: Path) -> dict[str, float | None]:
    """Fold means of one 10-fold run."""
    folds = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(run.glob("fold-*.json"))]
    if len(folds) != 10:
        raise ValueError(f"{run} has {len(folds)} finished folds, expected 10")

    def mean(values: list[float | None]) -> float | None:
        present = [value for value in values if value is not None]
        return float(np.mean(present)) if present else None

    rows = []
    for fold in folds:
        test = fold["test"]
        accuracy = _horizon_mean(test, "accuracy")
        train = fold["modality_evaluations"]["full"]["train"].get("classification_accuracy")
        rows.append({
            "accuracy": accuracy,
            "precision_macro": _horizon_mean(test, "precision_macro"),
            "mse": _horizon_mean(test, "mse"),
            "mae": _horizon_mean(test, "mae"),
            "train_test_gap": None if train is None else train - test["classification_accuracy"],
            "persistence": test.get("baselines", {}).get("persistence_class_accuracy"),
            "accuracy_label_changed": test.get("photo_change", {}).get("accuracy_label_changed"),
            "train_windows": fold["window_counts"]["train"],
            "test_windows": fold["window_counts"]["test"],
        })
    return {key: mean([row[key] for row in rows]) for key in METRICS}


def analyse(root: Path) -> dict[str, Any]:
    seeds = sorted(path for path in root.iterdir() if (path / "splits").is_dir())
    per_seed: dict[str, dict[str, dict[str, float | None]]] = {}
    for seed in seeds:
        per_seed[seed.name] = {
            arm: run_metrics(seed / "splits" / arm)
            for arm in ARMS if (seed / "splits" / arm / "cross-validation-results.json").is_file()
        }
    def spread(values: list[float]) -> dict[str, Any]:
        return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "seeds": len(values)}

    summary: dict[str, Any] = {}
    reference = ARMS[0]
    for arm in ARMS:
        values: dict[str, list[float]] = {key: [] for key in METRICS}
        differences: dict[str, list[float]] = {key: [] for key in METRICS}
        for runs in per_seed.values():
            for key in METRICS:
                value = runs[arm][key] if arm in runs else None
                if value is None:
                    continue
                values[key].append(value)
                base = runs[reference][key] if reference in runs else None
                if base is not None:
                    differences[key].append(value - base)
        summary[arm] = {key: spread(v) for key, v in values.items() if v}
        if arm != reference:
            summary[arm]["difference_to_leave_one_tool_out"] = {key: spread(v) for key, v in differences.items() if v}
    paper = {key: float(np.mean(values)) for key, values in PAPER_OMNIFUSER_RESULTS.items()}
    return {"seeds": list(per_seed), "per_seed": per_seed, "summary": summary, "paper": paper}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("experiment_results/splits"))
    args = parser.parse_args()
    result = analyse(args.root)
    output = args.root / "split-study.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"seeds: {', '.join(result['seeds'])}")
    print(f"{'split':20s} {'accuracy':>15s} {'precision':>15s} {'MSE':>15s} {'MAE':>15s} {'train-test gap':>15s}")

    def cell(entry: dict[str, Any] | None) -> str:
        if not entry:
            return f"{'n/a':>15s}"
        sd = entry.get("sd")
        return f"{entry['mean']:8.3f} ± {sd:.3f}" if sd is not None else f"{entry['mean']:15.3f}"

    for arm, entry in result["summary"].items():
        print(f"{arm:20s} " + " ".join(cell(entry.get(k)) for k in ("accuracy", "precision_macro", "mse", "mae", "train_test_gap")))
    paper = result["paper"]
    print(f"{'paper (OmniFuser)':20s} {paper['accuracy']:15.3f} {paper['precision']:15.3f} {paper['mse']:15.3f} {paper['mae']:15.3f}")
    for arm, entry in result["summary"].items():
        if "difference_to_leave_one_tool_out" in entry:
            d = entry["difference_to_leave_one_tool_out"]
            print(f"{arm} - leave-one-tool-out: accuracy {d['accuracy']['mean']:+.3f}, MSE {d['mse']['mean']:+.3f}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
