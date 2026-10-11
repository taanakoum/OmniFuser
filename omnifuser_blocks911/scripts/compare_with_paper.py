"""Put the scores of a finished run next to the OmniFuser rows of the paper's Tables IV and V (Mudestreda).

    python scripts/compare_with_paper.py runs/lag1_pool1000
    python scripts/compare_with_paper.py runs/lag1_pool1000 --average weighted --by pooled_by_windows --json table.json

Reads every ``fold-*.json`` written by ``train_fold`` / ``run_cross_validation`` in RUN_DIR (the ``paper_horizons`` and
``baselines`` entries) and prints, for each horizon 12 ... 96:

  MSE, MAE            z-scored force units, cumulative over leads 1..h (the paper's Table IV unit)
  last-value          MSE of "repeat the last observed force"; a model that does not beat it has learned nothing
                      beyond persistence
  Acc, Pre            accuracy and precision over leads 1..h (Table V); Pre is macro- or class-weighted (--average),
                      the paper does not say which
  persistence         accuracy of "the label of the photo shown at the last input step stays"

The paper's numbers are quoted from Tables IV and V of arXiv 2511.01320v1 (OmniFuser, Mudestreda).  They come from a
protocol that the paper does not describe fully (windows, step size, normalisation, split, one model per horizon or
not), so the comparison is a plausibility check: the trend over h and the position relative to the baselines mean more
than the third decimal.  Horizons larger than the number of leads the model was trained with are not reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

HORIZONS = (12, 24, 36, 48, 60, 72, 84, 96)
PAPER_OMNIFUSER = {  # Tables IV (MSE, MAE) and V (Acc, Pre), OmniFuser, Mudestreda, horizons 12 ... 96
    "mse": (0.467, 0.514, 0.586, 0.686, 0.772, 0.830, 0.917, 0.902),
    "mae": (0.475, 0.509, 0.548, 0.590, 0.630, 0.654, 0.686, 0.693),
    "accuracy": (0.886, 0.928, 0.838, 0.866, 0.820, 0.863, 0.869, 0.863),
    "precision": (0.890, 0.922, 0.905, 0.913, 0.802, 0.862, 0.853, 0.842),
}
MODEL_KEYS = ("mse", "mae", "accuracy", "precision_macro", "recall_macro", "f1_macro",
              "precision_weighted", "recall_weighted", "f1_weighted")
BASELINE_KEYS = ("last_value_mse", "last_value_mae", "zero_prediction_mse", "persistence_class_accuracy")


def load_folds(run_dir: str | Path) -> list[dict[str, Any]]:
    paths = sorted(Path(run_dir).glob("fold-*.json"))
    if not paths:
        raise FileNotFoundError(f"no fold-*.json in {run_dir}")
    folds = []
    for path in paths:
        with path.open(encoding="utf-8") as file:
            folds.append(json.load(file))
    return folds


def aggregate(folds: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Per horizon: the unweighted mean over folds and the average weighted by the number of test windows."""
    horizons = sorted({h for fold in folds for h in fold["test"].get("paper_horizons", {})}, key=int)
    table: dict[str, dict[str, dict[str, float]]] = {"mean_over_folds": {}, "pooled_by_windows": {}}
    for horizon in horizons:
        rows = []
        for fold in folds:
            model = fold["test"].get("paper_horizons", {}).get(horizon)
            if model is None:
                continue
            base = fold["test"].get("baselines", {}).get("by_horizon", {}).get(horizon, {})
            rows.append(({**model, **base}, float(fold.get("window_counts", {}).get("test", 1))))
        keys = sorted({key for row, _ in rows for key in row})
        mean, pooled = {}, {}
        for key in keys:
            have = [(row[key], weight) for row, weight in rows if key in row]
            values, weights = np.array([v for v, _ in have]), np.array([w for _, w in have])
            mean[key] = float(values.mean())
            pooled[key] = float((values * weights).sum() / weights.sum())
        table["mean_over_folds"][horizon], table["pooled_by_windows"][horizon] = mean, pooled
    return table


def _cell(value: float | None) -> str:
    return "   -  " if value is None else f"{value:6.3f}"


def render(table: dict[str, dict[str, dict[str, float]]], by: str = "mean_over_folds", average: str = "macro") -> str:
    precision = f"precision_{average}"
    lines = [f"{'h':>3} | {'MSE':>6} {'last-value':>10} {'paper':>6} | {'MAE':>6} {'paper':>6} | "
             f"{'Acc':>6} {'persistence':>11} {'paper':>6} | {'Pre':>6} {'paper':>6}",
             "-" * 96]
    beats_mse = beats_acc = shown = 0
    for horizon, row in table[by].items():
        index = HORIZONS.index(int(horizon)) if int(horizon) in HORIZONS else None
        paper = {k: (PAPER_OMNIFUSER[k][index] if index is not None else None) for k in PAPER_OMNIFUSER}
        lines.append(
            f"{int(horizon):>3} | {_cell(row.get('mse'))} {_cell(row.get('last_value_mse')):>10} {_cell(paper['mse'])} | "
            f"{_cell(row.get('mae'))} {_cell(paper['mae'])} | "
            f"{_cell(row.get('accuracy'))} {_cell(row.get('persistence_class_accuracy')):>11} {_cell(paper['accuracy'])} | "
            f"{_cell(row.get(precision))} {_cell(paper['precision'])}")
        if "last_value_mse" in row:
            shown += 1
            beats_mse += row["mse"] < row["last_value_mse"]
            beats_acc += row["accuracy"] > row.get("persistence_class_accuracy", 1.0)
    lines.append("")
    if shown:
        lines.append(f"model MSE below last-value MSE at {beats_mse} of {shown} horizons; "
                     f"accuracy above persistence at {beats_acc} of {shown} horizons.")
    lines.append("MSE, MAE in z-scored units, cumulative over leads 1..h; paper = OmniFuser on Mudestreda, "
                 "Tables IV and V.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--by", choices=("mean_over_folds", "pooled_by_windows"), default="mean_over_folds")
    parser.add_argument("--average", choices=("macro", "weighted"), default="macro",
                        help="averaging of the precision column (the paper does not say)")
    parser.add_argument("--json", help="write the aggregated table to this file")
    args = parser.parse_args()
    folds = load_folds(args.run_dir)
    first = folds[0]
    print(f"{len(folds)} folds ({', '.join(f['test_tool'] for f in folds)}); image_lag = {first.get('image_lag')}, "
          f"pool = {first.get('pool')}, label column = {first.get('label_column')}, stride = {first.get('stride')}, "
          f"model horizon = {first.get('config', {}).get('horizon')}")
    table = aggregate(folds)
    print(render(table, args.by, args.average))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump(table, file, indent=2)


if __name__ == "__main__":
    main()
