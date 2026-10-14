"""Compare trivial forecasting/classification baselines at candidate time steps.

Examples:
    python scripts/task_diagnostics.py --data-dir Nonastreda_Data
    python scripts/task_diagnostics.py --data-dir Nonastreda_Data --pools 1 10 100 1000 --horizons 12 24 96

Normalization is computed from the pooled diagnostic records (including all tools).
This is descriptive only; model training fits normalization on training tools.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from omnifuser.data import Record, load_nonastreda, make_tool_timelines, pool_records


def window_starts(
    length: int,
    input_len: int,
    horizon: int,
    stride: int,
    limit: int | None,
) -> np.ndarray:
    if input_len < 1 or horizon < 1 or stride < 1:
        raise ValueError("input length, horizon, and stride must be positive")
    if limit is not None and limit < 1:
        raise ValueError("window limit must be positive")
    starts = np.arange(0, length - input_len - horizon + 1, stride)
    if limit is not None and len(starts) > limit:
        starts = starts[:: math.ceil(len(starts) / limit)]
    return starts


def fit_diagnostic_normalizer(
    records: Sequence[Record],
) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("cannot normalize an empty record set")
    values = np.concatenate([record.force for record in records]).astype(
        np.float64, copy=False
    )
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def measure(
    timelines: Sequence[Any],
    mean: np.ndarray,
    scale: np.ndarray,
    input_len: int,
    horizon: int,
    limit: int | None,
) -> dict[str, float | int]:
    squared_last = squared_zero = 0.0
    elements = windows = changed = persistence_hits = label_count = 0
    for timeline in timelines:
        force = (timeline.force - mean) / scale
        starts = window_starts(len(force), input_len, horizon, horizon, limit)
        if not len(starts):
            continue
        last = force[starts + input_len - 1]
        shown_labels = (
            timeline.image_labels
            if timeline.image_labels is not None
            else timeline.labels
        )[starts + input_len - 1]
        target_labels = np.stack(
            [
                timeline.labels[starts + input_len + lead]
                for lead in range(horizon)
            ],
            axis=1,
        )
        for lead in range(horizon):
            target = force[starts + input_len + lead]
            squared_last += float(np.square(target - last).sum())
            squared_zero += float(np.square(target).sum())
        elements += len(starts) * horizon * force.shape[1]
        windows += len(starts)
        changed += int((target_labels != target_labels[:, :1]).any(axis=1).sum())
        persistence_hits += int(
            (target_labels == shown_labels[:, None]).sum()
        )
        label_count += target_labels.size
    return {
        "windows": windows,
        "zero_mse": squared_zero / max(elements, 1),
        "last_value_mse": squared_last / max(elements, 1),
        "label_change_pct": 100.0 * changed / max(windows, 1),
        "persistence_acc": persistence_hits / max(label_count, 1),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Nonastreda_Data"))
    parser.add_argument("--pools", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument("--horizons", type=int, nargs="+", default=[12, 24, 96])
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--fs", type=float, default=1000.0)
    parser.add_argument("--max-windows-per-tool", type=int, default=20000)
    parser.add_argument(
        "--trim-pass", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.fs <= 0 or any(value < 1 for value in args.pools + args.horizons):
        parser.error("sampling rate, pools, and horizons must be positive")
    if args.input_len < 1 or args.max_windows_per_tool < 1:
        parser.error("input length and max windows per tool must be positive")

    records = load_nonastreda(args.data_dir, trim_pass=args.trim_pass)
    rows: list[dict[str, float | int]] = []
    print(
        f"{'pool':>5} {'step':>8} {'P':>4} {'lag':>3} "
        f"{'windows':>8} {'zero':>7} {'last':>7} {'chg %':>7} {'persist':>8}"
    )
    for factor in args.pools:
        pooled_records = pool_records(records, factor)
        mean, scale = fit_diagnostic_normalizer(pooled_records)
        step_seconds = factor / args.fs
        for lag in (0, 1):
            timelines = make_tool_timelines(pooled_records, image_lag=lag)
            for horizon in args.horizons:
                result = measure(
                    timelines,
                    mean,
                    scale,
                    args.input_len,
                    horizon,
                    args.max_windows_per_tool,
                )
                row: dict[str, float | int] = {
                    "pool": factor,
                    "step_seconds": step_seconds,
                    "horizon": horizon,
                    "image_lag": lag,
                    **result,
                }
                rows.append(row)
                print(
                    f"{factor:5d} {step_seconds:7.3f}s {horizon:4d} {lag:3d} "
                    f"{result['windows']:8d} {result['zero_mse']:7.3f} "
                    f"{result['last_value_mse']:7.3f} "
                    f"{result['label_change_pct']:7.2f} "
                    f"{result['persistence_acc']:8.4f}"
                )
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
