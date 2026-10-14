"""Compare finished fold JSON files with the OmniFuser paper's Tables IV-V.

Example:
    python scripts/compare_with_paper.py runs/a_lag1_pool1000_direct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from omnifuser.evaluation import PAPER_HORIZONS as HORIZONS
from omnifuser.evaluation import PAPER_OMNIFUSER_RESULTS as PAPER


def load_folds(run_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(run_dir.glob("fold-*.json"))
    if not paths:
        raise FileNotFoundError(f"no fold-*.json files found in {run_dir}")
    folds = []
    for path in paths:
        with path.open(encoding="utf-8") as file:
            folds.append(json.load(file))
    return folds


def aggregate(folds: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    horizons = sorted(
        {
            horizon
            for fold in folds
            for horizon in fold["test"].get("paper_horizons", {})
        },
        key=int,
    )
    aggregate: dict[str, dict[str, dict[str, float]]] = {
        "mean_over_folds": {},
        "pooled_by_windows": {},
    }
    for horizon in horizons:
        rows: list[tuple[dict[str, float], float]] = []
        for fold in folds:
            metrics = fold["test"].get("paper_horizons", {}).get(horizon)
            if metrics is None:
                continue
            baseline = fold["test"].get("baselines", {}).get("by_horizon", {}).get(
                horizon, {}
            )
            weight = float(fold.get("window_counts", {}).get("test", 1))
            rows.append(({**metrics, **baseline}, weight))
        keys = sorted({key for row, _ in rows for key in row})
        aggregate["mean_over_folds"][horizon] = {
            key: float(np.mean([row[key] for row, _ in rows if key in row]))
            for key in keys
        }
        aggregate["pooled_by_windows"][horizon] = {
            key: float(
                np.average(
                    [row[key] for row, _ in rows if key in row],
                    weights=[weight for row, weight in rows if key in row],
                )
            )
            for key in keys
        }
    return aggregate


def _format(value: float | None) -> str:
    return "   -" if value is None else f"{value:6.3f}"


def render(
    table: dict[str, dict[str, dict[str, float]]],
    method: str,
    average: str,
) -> str:
    metric = table[method]
    precision_key = f"precision_{average}"
    lines = [
        "  h |    MSE   last-value   paper |    MAE  paper |    Acc persistence  paper |    Pre  paper",
        "-" * 96,
    ]
    for horizon, row in metric.items():
        index = HORIZONS.index(int(horizon)) if int(horizon) in HORIZONS else None
        paper = {
            name: values[index] if index is not None else None
            for name, values in PAPER.items()
        }
        lines.append(
            f"{int(horizon):3} | {_format(row.get('mse'))} "
            f"{_format(row.get('last_value_mse')):>11} {_format(paper['mse'])} | "
            f"{_format(row.get('mae'))} {_format(paper['mae'])} | "
            f"{_format(row.get('accuracy'))} "
            f"{_format(row.get('persistence_class_accuracy')):>11} "
            f"{_format(paper['accuracy'])} | {_format(row.get(precision_key))} "
            f"{_format(paper['precision'])}"
        )
    lines.append(
        "MSE/MAE are z-scored cumulative metrics. Paper values are OmniFuser on "
        "Mudestreda (Tables IV-V); Nonastreda scores are not directly comparable."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--by",
        choices=("mean_over_folds", "pooled_by_windows"),
        default="mean_over_folds",
    )
    parser.add_argument("--average", choices=("macro", "weighted"), default="macro")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    folds = load_folds(args.run_dir)
    first = folds[0]
    print(
        f"{len(folds)} fold(s); image_lag={first.get('image_lag')}, "
        f"pool={first.get('pool')}, label={first.get('label_column')}, "
        f"stride={first.get('stride')}, horizon={first.get('config', {}).get('horizon')}"
    )
    table = aggregate(folds)
    print(render(table, args.by, args.average))
    if args.json:
        with args.json.open("w", encoding="utf-8") as file:
            json.dump(table, file, indent=2)


if __name__ == "__main__":
    main()
