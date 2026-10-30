"""Re-score a finished run's saved fold models with forecasts at every phase of a pass.

The evaluation windows of every run start every 96 steps, about one pass (median 98.5 s),
so forecasts begin near pass boundaries and their first leads hold the entry and exit
transients (FINAL_REPORT_REPRODUCTION.md §6.3). This script re-predicts each fold's test
tool with its saved model (``fold-Tn.pt``, no training) at other strides, e.g. 24, whose
windows start at every phase of a pass, and compares the per-lead and cumulative
z-scored MSE. If the falling cumulative curve comes from the phase-locking, it should
rise with the horizon at stride 24.

Needs the ``fold-Tn.pt`` files and the data (run it on Ruche, or after unpacking
``results-checkpoints.tar.gz``).

Example:
    python scripts/phase_study.py experiment_results/main/full-model --data-dir Nonastreda_Data --strides 96 24
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from omnifuser.evaluation import PAPER_HORIZONS
from omnifuser.model_bundle import predict

FOLDS = tuple(f"T{index}" for index in range(1, 11))


def per_lead_mse(prediction_file: Path, scale: np.ndarray) -> np.ndarray:
    with np.load(prediction_file) as arrays:
        error = (arrays["force_predictions"] - arrays["force_targets"]) / scale
    return np.square(error).mean(axis=(0, 2))


def study(run: Path, data_dir: Path, strides: list[int], device: str) -> dict[str, Any]:
    output = run / "phase-study"
    curves: dict[str, Any] = {}
    for stride in strides:
        leads: list[np.ndarray] = []
        windows: dict[str, int] = {}
        for fold in FOLDS:
            scale = np.asarray(json.loads((run / f"fold-{fold}.json").read_text(encoding="utf-8"))["normalizer_scale"])
            prediction_file = output / f"predictions-{fold}-stride{stride}.npz"
            if not prediction_file.is_file():
                metrics = predict(run / f"fold-{fold}.pt", data_dir, fold, output,
                                  device_name=device, evaluation_stride=stride)
                windows[fold] = metrics["prediction"]["windows"]
            leads.append(per_lead_mse(prediction_file, scale))
        fold_mean = np.mean(leads, axis=0)  # folds weighted equally, as in the reports
        cumulative = np.cumsum(fold_mean) / np.arange(1, len(fold_mean) + 1)
        curves[str(stride)] = {
            "per_lead_mse": fold_mean.tolist(),
            "cumulative_mse": {str(h): float(cumulative[h - 1]) for h in PAPER_HORIZONS if h <= len(cumulative)},
            "windows_per_fold": windows,
        }
    return curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="a run folder with fold-Tn.pt and fold-Tn.json files")
    parser.add_argument("--data-dir", type=Path, default=Path("Nonastreda_Data"))
    parser.add_argument("--strides", type=int, nargs="+", default=[96, 24])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    curves = study(args.run, args.data_dir, args.strides, args.device)
    (args.run / "phase-study" / "phase-study.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
    print("cumulative z-scored MSE over leads 1..h (fold mean)")
    print("stride " + " ".join(f"{h:>6d}" for h in PAPER_HORIZONS))
    for stride, curve in curves.items():
        print(f"{stride:>6s} " + " ".join(f"{curve['cumulative_mse'][str(h)]:6.3f}" for h in PAPER_HORIZONS))
    print(f"wrote {args.run / 'phase-study' / 'phase-study.json'}")


if __name__ == "__main__":
    main()
