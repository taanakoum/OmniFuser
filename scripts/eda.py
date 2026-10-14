"""Exploratory data analysis of the Nonastreda inputs used by the OmniFuser pipeline.

Writes tables, a JSON summary, and figures to one folder (default
``experiment_results/eda``). ``EDA.md`` explains every metric.

    python scripts/eda.py --data-dir Nonastreda_Data --out experiment_results/eda
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr

from omnifuser.data import (
    CLASS_NAMES,
    PASS_TRIM_REFERENCE_FRACTION,
    PASS_TRIM_THRESHOLD_FRACTION,
    PASS_TRIM_WINDOW_SAMPLES,
    ForceNormalizer,
    ToolTimeline,
    _nonastreda_span_bounds,
    leave_one_tool_out_folds,
    load_nonastreda,
    make_tool_timelines,
)
from omnifuser.evaluation import PAPER_HORIZONS, PAPER_OMNIFUSER_RESULTS

CHANNELS = ("Fx", "Fy", "Fz")
# Reference palette slots 1-3 (validated all-pairs for colour-vision deficiency).
CLASS_COLORS = {"sharp": "#2a78d6", "used": "#eb6834", "dulled": "#1baf7a"}
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------------- helpers
def _decode(record_id: str) -> tuple[str, int, int]:
    stem = Path(record_id).stem
    tool, rest = stem[1:].split("R")
    run, block = rest.split("B")
    return f"T{int(tool)}", int(run), int(block)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def window_starts(length: int, span: int, stride: int, limit: int | None = None) -> np.ndarray:
    if length < span:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, length - span + 1, stride, dtype=np.int64)
    if limit is not None and len(starts) > limit:
        starts = starts[:: math.ceil(len(starts) / limit)]
    return starts


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    centered = series - series.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        return np.full(max_lag + 1, np.nan)
    return np.array(
        [1.0] + [float(np.dot(centered[:-lag], centered[lag:]) / denominator) for lag in range(1, max_lag + 1)]
    )


# ----------------------------------------------------------------------------- raw records
def trimming_diagnostics(force: np.ndarray) -> dict[str, Any]:
    """Re-derive the loader's span trim and flag records whose reference window is idle.

    The reference level is the median envelope of the first 5 % of the pass. If more
    than half of that window lies below half the pass median (> ~2.5 % of the pass),
    the reference is a low level and the start is not trimmed. "Idle" here means
    "below half the pass median": it also matches low-force cutting ramps, so flagged
    passes need visual review (figure 05 and EDA.md). ``residual_*`` counts the
    low-level samples the trim leaves at either end.
    """
    start, end = _nonastreda_span_bounds(force)
    n = len(force)
    magnitude = np.square(force.astype(np.float64)).sum(axis=1)
    envelope = np.sqrt(uniform_filter1d(magnitude, size=PASS_TRIM_WINDOW_SAMPLES, mode="nearest"))
    cutting_level = float(np.median(envelope))
    idle = envelope < PASS_TRIM_THRESHOLD_FRACTION * cutting_level
    reference = max(PASS_TRIM_WINDOW_SAMPLES, int(n * PASS_TRIM_REFERENCE_FRACTION))
    active = np.flatnonzero(~idle)
    lead_idle = int(active[0]) if active.size else n
    runout_idle = int(n - 1 - active[-1]) if active.size else n
    retained = idle[start:end]
    alt_start, alt_end = (int(active[0]), int(active[-1]) + 1) if active.size else (0, n)
    return {
        "trim_start": start,
        "trim_end": end,
        "trimmed_lead_samples": start,
        "trimmed_runout_samples": n - end,
        "cutting_level": cutting_level,
        "reference_level": float(np.median(envelope[:reference])),
        "reference_idle_fraction": float(idle[:reference].mean()),
        "reference_is_idle": bool(idle[:reference].mean() > 0.5),
        "lead_idle_samples": lead_idle,
        "runout_idle_samples": runout_idle,
        "residual_lead_idle_samples": max(0, lead_idle - start),
        "residual_runout_idle_samples": max(0, runout_idle - (n - end)),
        "retained_idle_fraction": float(retained.mean()) if retained.size else 0.0,
        "alternative_trim_lead_samples": alt_start,
        "alternative_trim_runout_samples": n - alt_end,
    }


def raw_record_table(data_dir: Path, fs: float) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    labels = {row["id"]: row for row in csv.DictReader((data_dir / "labels.csv").open(encoding="utf-8-sig"))}
    regression = {row["id"]: row for row in csv.DictReader((data_dir / "labels_reg.csv").open(encoding="utf-8-sig"))}
    datastore = loadmat(data_dir / "forces_xyz_raw.mat", variable_names=["baseDatastore"], simplify_cells=True)[
        "baseDatastore"
    ]
    rows: list[dict[str, Any]] = []
    examples: dict[str, np.ndarray] = {}
    for entry in datastore:
        record_id = Path(str(entry[0])).stem
        tool, run, block = _decode(record_id)
        force = np.asarray(entry[3], dtype=np.float64).T
        image_path = data_dir / "tool" / f"{record_id}.jpg"
        magnitude = np.sqrt(np.square(force).sum(axis=1))
        row: dict[str, Any] = {
            "record_id": record_id,
            "tool": tool,
            "run": run,
            "block": block,
            "image_label": labels[record_id]["image_label"].strip().lower(),
            "mat_label": str(entry[1]).strip().lower(),
            "tool_label": labels[record_id]["tool_label"].strip(),
            "flank_wear": float(regression[record_id]["flank_wear"]),
            "gaps": float(regression[record_id]["gaps"]),
            "overhang": float(regression[record_id]["overhang"]),
            "samples": len(force),
            "duration_s": len(force) / fs,
            "non_finite_values": int((~np.isfinite(force)).sum()),
            "magnitude_mean": float(magnitude.mean()),
            "magnitude_rms": float(np.sqrt(np.square(magnitude).mean())),
            "magnitude_sum": float(magnitude.sum()),
            "image_bytes": image_path.stat().st_size if image_path.is_file() else None,
        }
        for index, channel in enumerate(CHANNELS):
            values = force[:, index]
            row.update(
                {
                    f"{channel}_mean": float(values.mean()),
                    f"{channel}_std": float(values.std()),
                    f"{channel}_min": float(values.min()),
                    f"{channel}_max": float(values.max()),
                }
            )
        row.update(trimming_diagnostics(force))
        row["_force_hash"] = hash(force.tobytes())
        rows.append(row)
        examples[record_id] = magnitude.astype(np.float32)
    rows.sort(key=lambda row: (int(row["tool"][1:]), row["run"], row["block"]))
    return rows, examples


def image_statistics(rows: list[dict[str, Any]], data_dir: Path) -> None:
    for row in rows:
        with Image.open(data_dir / "tool" / f"{row['record_id']}.jpg") as image:
            row["image_width"], row["image_height"] = image.size
            image.draft("RGB", (image.size[0] // 8, image.size[1] // 8))
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        luminance = pixels @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        row["image_mean_luminance"] = float(luminance.mean())
        row["image_std_luminance"] = float(luminance.std())
        gradient = np.abs(np.diff(luminance, axis=0)).mean() + np.abs(np.diff(luminance, axis=1)).mean()
        row["image_edge_strength"] = float(gradient)


def duplicate_analysis(rows: list[dict[str, Any]], retained_ids: set[str]) -> dict[str, Any]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["tool"], row["run"], row["_force_hash"])].append(row)
    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    for row in rows:
        row["retained_after_dedup"] = row["record_id"] in retained_ids
    pattern = Counter("+".join(f"B{row['block']}" for row in group) for group in duplicate_groups)
    label_conflicts = sum(len({row["image_label"] for row in group}) > 1 for group in duplicate_groups)
    return {
        "raw_records": len(rows),
        "retained_records": len(retained_ids),
        "duplicate_groups": len(duplicate_groups),
        "records_removed": len(rows) - len(retained_ids),
        "duplicate_block_patterns": dict(pattern),
        "duplicate_groups_with_different_image_labels": label_conflicts,
    }


def label_dynamics(rows: list[dict[str, Any]], subset: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Class mixing within runs and backward (healing) class flips along tool order."""
    index = {name: position for position, name in enumerate(CLASS_NAMES)}
    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tool[row["tool"]].append(row)
    runs: list[dict[str, Any]] = []
    backward = forward = 0
    flank_decreases = 0
    for tool, records in by_tool.items():
        records.sort(key=lambda row: (row["run"], row["block"]))
        sequence = [index[row["image_label"]] for row in records]
        backward += sum(b < a for a, b in pairwise(sequence))
        forward += sum(b > a for a, b in pairwise(sequence))
        flank_decreases += sum(
            b["flank_wear"] < a["flank_wear"] for a, b in pairwise(records)
        )
        by_run: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            by_run[row["run"]].append(row)
        for run, members in sorted(by_run.items()):
            labels = [row["image_label"] for row in members]
            runs.append(
                {
                    "subset": subset,
                    "tool": tool,
                    "run": run,
                    "blocks": "|".join(str(row["block"]) for row in members),
                    "labels": "|".join(labels),
                    "mixed_classes": len(set(labels)) > 1,
                    "flank_wear": "|".join(f"{row['flank_wear']:.1f}" for row in members),
                }
            )
    summary = {
        "runs": len(runs),
        "runs_with_mixed_classes": sum(run["mixed_classes"] for run in runs),
        "backward_class_flips": backward,
        "forward_class_flips": forward,
        "flank_wear_decreases_along_tool_order": flank_decreases,
    }
    return summary, runs


# ----------------------------------------------------------------------------- windows and folds
def _spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    result: Any = spearmanr(x, y)  # SciPy's result object has no usable stubs
    return {"rho": float(result.statistic), "p_value": float(result.pvalue)}


def _shown_labels(timeline: ToolTimeline) -> np.ndarray:
    """Label of the photo shown at each step (the step's own label if not tracked)."""
    return timeline.image_labels if timeline.image_labels is not None else timeline.labels


def _record_codes(timeline: ToolTimeline) -> np.ndarray:
    return timeline.cycle_codes.astype(np.int64) * 100 + timeline.batch_codes


def protocol_windows(
    data_dir: Path, pool: int, input_len: int, horizon: int, stride: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[int, list[ToolTimeline]]]:
    records = load_nonastreda(data_dir, pool=pool)
    timelines = {lag: make_tool_timelines(records, image_lag=lag) for lag in (0, 1)}
    eval_stride = max(stride, horizon)
    span = input_len + horizon
    tool_rows: list[dict[str, Any]] = []
    for timeline, lagged in zip(timelines[0], timelines[1]):
        codes = _record_codes(timeline)
        starts = window_starts(len(timeline.force), span, eval_stride)
        persistence = {0: [], 1: []}
        change, multi_record, images = [], [], []
        for start in starts:
            end_input = start + input_len
            target = timeline.labels[end_input : end_input + horizon]
            change.append(bool((target != timeline.labels[end_input - 1]).any()))
            multi_record.append(len(np.unique(codes[start:end_input])) > 1)
            images.append(len(np.unique(lagged.image_codes[start:end_input])))
            persistence[0].append(float((target == _shown_labels(timeline)[end_input - 1]).mean()))
            persistence[1].append(float((target == _shown_labels(lagged)[end_input - 1]).mean()))
        counts = Counter(timeline.labels.tolist())
        tool_rows.append(
            {
                "tool": timeline.tool,
                "records": len(np.unique(codes)),
                "steps": len(timeline.force),
                **{f"steps_{name}": counts.get(position, 0) for position, name in enumerate(CLASS_NAMES)},
                "train_windows_stride": len(window_starts(len(timeline.force), span, stride)),
                "eval_windows": len(starts),
                "eval_windows_target_class_change": float(np.mean(change)) if change else None,
                "eval_windows_input_spans_records": float(np.mean(multi_record)) if multi_record else None,
                "eval_windows_mean_unique_images_lag1": float(np.mean(images)) if images else None,
                "persistence_accuracy_lag0": float(np.mean(persistence[0])) if starts.size else None,
                "persistence_accuracy_lag1": float(np.mean(persistence[1])) if starts.size else None,
            }
        )

    fold_rows: list[dict[str, Any]] = []
    by_tool = {timeline.tool: timeline for timeline in timelines[1]}
    for number, fold in enumerate(leave_one_tool_out_folds(list(by_tool)), start=1):
        train = [by_tool[tool] for tool in fold.train_tools]
        normalizer = ForceNormalizer.fit(train)
        row: dict[str, Any] = {
            "fold": number,
            "test_tool": fold.test_tools[0],
            "validation_tool": fold.validation_tools[0],
            "train_tools": " ".join(fold.train_tools),
        }
        for split, tools, split_stride in (
            ("train", fold.train_tools, stride),
            ("validation", fold.validation_tools, eval_stride),
            ("test", fold.test_tools, eval_stride),
        ):
            targets: list[np.ndarray] = []
            for tool in tools:
                timeline = by_tool[tool]
                for start in window_starts(len(timeline.force), span, split_stride):
                    targets.append(timeline.labels[start + input_len : start + span])
            stacked = np.concatenate(targets) if targets else np.empty(0, dtype=np.int8)
            row[f"{split}_windows"] = len(targets)
            for position, name in enumerate(CLASS_NAMES):
                row[f"{split}_target_share_{name}"] = float((stacked == position).mean()) if stacked.size else None
        for index, channel in enumerate(CHANNELS):
            row[f"normalizer_mean_{channel}"] = float(normalizer.mean[index])
            row[f"normalizer_scale_{channel}"] = float(normalizer.scale[index])
            test = by_tool[fold.test_tools[0]].force[:, index]
            row[f"test_z_mean_{channel}"] = float((test.mean() - normalizer.mean[index]) / normalizer.scale[index])
            row[f"test_z_std_{channel}"] = float(test.std() / normalizer.scale[index])
        fold_rows.append(row)

    totals = {
        "pool": pool,
        "input_len": input_len,
        "horizon": horizon,
        "train_stride": stride,
        "evaluation_stride": eval_stride,
        "tool_steps_total": sum(row["steps"] for row in tool_rows),
        "eval_windows_total": sum(row["eval_windows"] for row in tool_rows),
        "train_windows_per_fold": _quantiles([row["train_windows"] for row in fold_rows]),
        "test_windows_per_fold": _quantiles([row["test_windows"] for row in fold_rows]),
    }
    return totals, tool_rows, fold_rows, timelines


def difficulty_by_pool(
    data_dir: Path, pools: Sequence[int], input_len: int, limit_per_tool: int
) -> list[dict[str, Any]]:
    """Trivial-baseline difficulty per time step, cumulative over the paper horizons."""
    horizon = max(PAPER_HORIZONS)
    span = input_len + horizon
    rows: list[dict[str, Any]] = []
    for pool in pools:
        records = load_nonastreda(data_dir, pool=pool)
        timelines = {lag: make_tool_timelines(records, image_lag=lag) for lag in (0, 1)}
        del records
        stacked = np.concatenate([timeline.force for timeline in timelines[0]]).astype(np.float64)
        mean, scale = stacked.mean(axis=0), stacked.std(axis=0)
        del stacked
        errors_last = {h: [] for h in PAPER_HORIZONS}
        errors_zero = {h: [] for h in PAPER_HORIZONS}
        abs_last = {h: [] for h in PAPER_HORIZONS}
        persistence = {(lag, h): [] for lag in (0, 1) for h in PAPER_HORIZONS}
        changed = {h: [] for h in PAPER_HORIZONS}
        windows = 0
        for timeline, lagged in zip(timelines[0], timelines[1]):
            z = (timeline.force - mean) / scale
            starts = window_starts(len(z), span, span, limit_per_tool)
            windows += len(starts)
            for start in starts:
                end_input = start + input_len
                last = z[end_input - 1]
                future = z[end_input : end_input + horizon]
                labels = timeline.labels[end_input : end_input + horizon]
                for h in PAPER_HORIZONS:
                    errors_last[h].append(np.square(future[:h] - last).mean())
                    abs_last[h].append(np.abs(future[:h] - last).mean())
                    errors_zero[h].append(np.square(future[:h]).mean())
                    changed[h].append(bool((labels[:h] != timeline.labels[end_input - 1]).any()))
                    persistence[(0, h)].append((labels[:h] == _shown_labels(timeline)[end_input - 1]).mean())
                    persistence[(1, h)].append((labels[:h] == _shown_labels(lagged)[end_input - 1]).mean())
        for index, h in enumerate(PAPER_HORIZONS):
            rows.append(
                {
                    "pool": pool,
                    "seconds_per_step": pool / 1000.0,
                    "horizon": h,
                    "windows": windows,
                    "last_value_mse": float(np.mean(errors_last[h])) if windows else None,
                    "last_value_mae": float(np.mean(abs_last[h])) if windows else None,
                    "zero_forecast_mse": float(np.mean(errors_zero[h])) if windows else None,
                    "target_label_change_share": float(np.mean(changed[h])) if windows else None,
                    "persistence_accuracy_lag0": float(np.mean(persistence[(0, h)])) if windows else None,
                    "persistence_accuracy_lag1": float(np.mean(persistence[(1, h)])) if windows else None,
                    "paper_omnifuser_mse": PAPER_OMNIFUSER_RESULTS["mse"][index],
                    "paper_omnifuser_accuracy": PAPER_OMNIFUSER_RESULTS["accuracy"][index],
                }
            )
        del timelines
    return rows


def acf_rows(timelines: list[ToolTimeline], raw_rows: list[dict[str, Any]], max_lag: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for timeline in timelines:
        for index, channel in enumerate(CHANNELS):
            values = autocorrelation(timeline.force[:, index].astype(np.float64), max_lag)
            rows.append({"level": "pooled_step", "tool": timeline.tool, "channel": channel,
                         **{f"lag_{lag}": float(value) for lag, value in enumerate(values)}})
    by_tool: dict[str, list[float]] = defaultdict(list)
    for row in raw_rows:
        if row["retained_after_dedup"]:
            by_tool[row["tool"]].append(row["magnitude_mean"])
    for tool, series in by_tool.items():
        lags = min(max_lag, len(series) - 2)
        values = autocorrelation(np.asarray(series), lags)
        rows.append({"level": "pass", "tool": tool, "channel": "magnitude_mean",
                     **{f"lag_{lag}": float(value) for lag, value in enumerate(values)}})

    def first_below(row: dict[str, Any], threshold: float) -> int | None:
        for lag in range(1, max_lag + 1):
            value = row.get(f"lag_{lag}")
            if value is not None and value < threshold:
                return lag
        return None

    pooled = [row for row in rows if row["level"] == "pooled_step"]
    passes = [row for row in rows if row["level"] == "pass"]
    summary = {
        "pooled_step_first_lag_below_0.5": _quantiles([first_below(r, 0.5) or max_lag + 1 for r in pooled]),
        "pooled_step_acf_at_lag_17": _quantiles([r["lag_17"] for r in pooled if "lag_17" in r]),
        "pass_level_first_lag_below_0.5": _quantiles([first_below(r, 0.5) or max_lag + 1 for r in passes]),
    }
    return rows, summary


# ----------------------------------------------------------------------------- figures
def _style(axis: Any, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_facecolor(SURFACE)
    axis.set_title(title, color=INK, fontsize=11, loc="left")
    axis.set_xlabel(xlabel, color=INK_MUTED)
    axis.set_ylabel(ylabel, color=INK_MUTED)
    axis.tick_params(colors=INK_MUTED, labelsize=8)
    axis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(GRID)


def make_figures(
    out: Path,
    raw_rows: list[dict[str, Any]],
    examples: dict[str, np.ndarray],
    tool_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    difficulty: list[dict[str, Any]],
    acf: list[dict[str, Any]],
    timelines: dict[int, list[ToolTimeline]],
    data_dir: Path,
    fs: float,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 9})
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    retained = [row for row in raw_rows if row["retained_after_dedup"]]
    tools = sorted({row["tool"] for row in raw_rows}, key=lambda tool: int(tool[1:]))

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(figures / name, dpi=150)
        plt.close(fig)
        written.append(f"figures/{name}")

    # 1. class counts per tool (retained records)
    fig, axis = plt.subplots(figsize=(8, 3.6))
    bottom = np.zeros(len(tools))
    for name in CLASS_NAMES:
        counts = np.array([sum(r["tool"] == t and r["image_label"] == name for r in retained) for t in tools])
        axis.bar(tools, counts, bottom=bottom, color=CLASS_COLORS[name], label=name, edgecolor=SURFACE, linewidth=1.5)
        bottom += counts
    _style(axis, "Retained records per tool and wear class", "tool", "records (passes)")
    axis.legend(frameon=False, ncol=3, loc="upper left")
    save(fig, "01_class_counts_by_tool.png")

    # 2. label sequence along tool order with flank wear
    fig, axes = plt.subplots(len(tools), 1, figsize=(9, 0.55 * len(tools) + 1), sharex=True)
    for axis, tool in zip(np.atleast_1d(axes), tools):
        records = sorted((r for r in retained if r["tool"] == tool), key=lambda r: (r["run"], r["block"]))
        for position, row in enumerate(records):
            axis.bar(position, 1, width=1, color=CLASS_COLORS[row["image_label"]], edgecolor=SURFACE, linewidth=0.5)
        axis.set_yticks([])
        axis.set_ylabel(tool, rotation=0, ha="right", va="center", color=INK_MUTED)
        for spine in axis.spines.values():
            spine.set_visible(False)
    np.atleast_1d(axes)[-1].set_xlabel("record position in (run, block) order", color=INK_MUTED)
    handles = [Rectangle((0, 0), 1, 1, color=CLASS_COLORS[name]) for name in CLASS_NAMES]
    fig.legend(handles, CLASS_NAMES, frameon=False, ncol=3, loc="upper right")
    fig.suptitle("Wear class of each retained pass, in tool order", x=0.01, ha="left", color=INK, fontsize=11)
    save(fig, "02_label_sequence_by_tool.png")

    # 3. flank wear per class
    fig, axis = plt.subplots(figsize=(6, 3.6))
    data = [[r["flank_wear"] for r in retained if r["image_label"] == name] for name in CLASS_NAMES]
    parts = axis.boxplot(data, tick_labels=list(CLASS_NAMES), patch_artist=True, widths=0.5)
    for patch, name in zip(parts["boxes"], CLASS_NAMES):
        patch.set_facecolor(CLASS_COLORS[name])
        patch.set_alpha(0.75)
    _style(axis, "Flank wear by image label (overlap = label ambiguity)", "image label", "flank wear (µm)")
    save(fig, "03_flank_wear_by_class.png")

    # 4. record length histogram
    fig, axis = plt.subplots(figsize=(6.5, 3.4))
    axis.hist([r["duration_s"] for r in raw_rows], bins=30, color=SERIES_COLORS[0], edgecolor=SURFACE)
    _style(axis, "Raw pass duration", f"seconds (at {fs:g} Hz)", "records")
    save(fig, "04_record_duration.png")

    # 5. trimming examples: largest residual idle lead-in and two typical passes
    flagged = [r for r in raw_rows if r["reference_is_idle"]][:4]
    typical = sorted(raw_rows, key=lambda r: abs(r["trimmed_lead_samples"] - np.median([x["trimmed_lead_samples"] for x in raw_rows])))[:1]
    chosen = flagged + typical
    fig, axes = plt.subplots(len(chosen), 1, figsize=(9, 2.0 * len(chosen)))
    for axis, row in zip(np.atleast_1d(axes), chosen):
        magnitude = examples[row["record_id"]]
        step = max(1, len(magnitude) // 4000)
        seconds = np.arange(0, len(magnitude), step) / fs
        axis.plot(seconds, magnitude[::step], color=SERIES_COLORS[0], linewidth=0.8)
        axis.axvspan(0, row["trim_start"] / fs, color=SERIES_COLORS[1], alpha=0.2, label="trimmed lead-in")
        axis.axvspan(row["trim_end"] / fs, len(magnitude) / fs, color=SERIES_COLORS[3], alpha=0.25, label="trimmed run-out")
        axis.axhline(row["reference_level"] * PASS_TRIM_THRESHOLD_FRACTION, color=INK_MUTED, linewidth=0.8, linestyle="--", label="trim threshold")
        flag = " - low reference window (review)" if row["reference_is_idle"] else ""
        _style(axis, f"{row['record_id']}: untrimmed low-level lead-in {row['residual_lead_idle_samples'] / fs:.1f} s{flag}", "seconds", "|F|")
    np.atleast_1d(axes)[0].legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
    save(fig, "05_trimming_examples.png")

    # 6. force level vs flank wear
    fig, axis = plt.subplots(figsize=(6.5, 4))
    for name in CLASS_NAMES:
        subset = [r for r in retained if r["image_label"] == name]
        axis.scatter([r["flank_wear"] for r in subset], [r["magnitude_mean"] for r in subset], s=18,
                     color=CLASS_COLORS[name], label=name, edgecolors=SURFACE, linewidths=0.6)
    _style(axis, "Mean force magnitude per pass vs flank wear", "flank wear (µm)", "mean |F|")
    axis.legend(frameon=False)
    save(fig, "06_force_vs_flank_wear.png")

    # 7. autocorrelation
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for axis, level, title in ((axes[0], "pooled_step", "Pooled 1 s steps, Fx, per tool"), (axes[1], "pass", "Pass-level mean |F|, per tool")):
        rows = [r for r in acf if r["level"] == level and r["channel"] in ("Fx", "magnitude_mean")]
        for row in rows:
            lags = sorted(int(key[4:]) for key in row if key.startswith("lag_"))
            axis.plot(lags, [row[f"lag_{lag}"] for lag in lags], color=SERIES_COLORS[0], alpha=0.35, linewidth=1.2)
        axis.axvline(17, color=SERIES_COLORS[1], linewidth=1.2, linestyle="--")
        axis.text(17.3, 0.9, "paper: decay beyond ~17", color=INK_MUTED, fontsize=8)
        axis.axhline(0, color=INK_MUTED, linewidth=0.6)
        _style(axis, title, "lag (steps)", "autocorrelation")
    save(fig, "07_autocorrelation.png")

    # 8. difficulty vs horizon by pool
    fig, axis = plt.subplots(figsize=(7.5, 4))
    pools = sorted({row["pool"] for row in difficulty})
    for color, pool in zip(SERIES_COLORS, pools):
        rows = [r for r in difficulty if r["pool"] == pool]
        axis.plot([r["horizon"] for r in rows], [r["last_value_mse"] for r in rows], color=color, linewidth=2, marker="o", markersize=4, label=f"last value, pool {pool}")
    axis.plot(PAPER_HORIZONS, PAPER_OMNIFUSER_RESULTS["mse"], color=INK, linewidth=2, linestyle="--", marker="s", markersize=4, label="paper OmniFuser (Table IV)")
    _style(axis, "Forecast difficulty: z-scored last-value MSE vs horizon", "horizon P (steps)", "cumulative MSE")
    axis.legend(frameon=False, fontsize=8)
    save(fig, "08_difficulty_by_time_step.png")

    # 9. persistence accuracy by lag
    fig, axis = plt.subplots(figsize=(7.5, 4))
    protocol_pool = 1000 if 1000 in pools else pools[-1]
    rows = [r for r in difficulty if r["pool"] == protocol_pool]
    axis.plot([r["horizon"] for r in rows], [r["persistence_accuracy_lag0"] for r in rows], color=SERIES_COLORS[1], linewidth=2, marker="o", label="label of shown photo, lag 0 (paper)")
    axis.plot([r["horizon"] for r in rows], [r["persistence_accuracy_lag1"] for r in rows], color=SERIES_COLORS[0], linewidth=2, marker="o", label="label of shown photo, lag 1 (protocol)")
    axis.plot(PAPER_HORIZONS, PAPER_OMNIFUSER_RESULTS["accuracy"], color=INK, linewidth=2, linestyle="--", marker="s", label="paper OmniFuser accuracy")
    axis.set_ylim(0, 1.02)
    _style(axis, f"Accuracy of copying the shown photo's label (pool {protocol_pool})", "horizon P (steps)", "cumulative accuracy")
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "09_persistence_by_image_lag.png")

    # 10. fold target-class shares
    fig, axis = plt.subplots(figsize=(9, 3.8))
    positions = np.arange(len(fold_rows))
    for offset, split in zip((-0.27, 0.0, 0.27), ("train", "validation", "test")):
        bottom = np.zeros(len(fold_rows))
        for name in CLASS_NAMES:
            share = np.array([row[f"{split}_target_share_{name}"] or 0.0 for row in fold_rows])
            axis.bar(positions + offset, share, width=0.25, bottom=bottom, color=CLASS_COLORS[name], edgecolor=SURFACE, linewidth=1,
                     label=name if split == "train" else None)
            bottom += share
    axis.set_xticks(positions, [f"{r['fold']}\ntest {r['test_tool']}" for r in fold_rows])
    _style(axis, "Target class shares per fold (bars: train | validation | test)", "fold", "share of target labels")
    axis.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    save(fig, "10_fold_class_shares.png")

    # 11. one tool stream after pooling
    timeline = max(timelines[0], key=lambda t: len(t.force))
    fig, axis = plt.subplots(figsize=(10, 3.4))
    steps = np.arange(len(timeline.force))
    axis.plot(steps, timeline.force[:, 0], color=INK_MUTED, linewidth=0.7)
    boundaries = np.flatnonzero(np.diff(_record_codes(timeline))) + 1
    edges = np.r_[0, boundaries, len(steps)]
    for left, right in pairwise(edges):
        axis.axvspan(left, right, color=CLASS_COLORS[CLASS_NAMES[timeline.labels[left]]], alpha=0.18, linewidth=0)
    handles = [Rectangle((0, 0), 1, 1, color=CLASS_COLORS[name], alpha=0.4) for name in CLASS_NAMES]
    axis.legend(handles, CLASS_NAMES, frameon=False, ncol=3, loc="upper left")
    _style(axis, f"{timeline.tool} force stream Fx after trimming and 1 s pooling (shading = pass label)", "pooled step", "Fx")
    save(fig, "11_tool_stream_example.png")

    # 12. image examples at model resolution
    fig, axes = plt.subplots(len(CLASS_NAMES), 3, figsize=(10, 4.2))
    for row_index, name in enumerate(CLASS_NAMES):
        subset = [r for r in retained if r["image_label"] == name]
        picks = [subset[int(i)] for i in np.linspace(0, len(subset) - 1, 3)]
        for column, row in enumerate(picks):
            with Image.open(data_dir / "tool" / f"{row['record_id']}.jpg") as image:
                pixels = image.convert("RGB").resize((384, 128), Image.Resampling.BILINEAR)
            axis = axes[row_index, column]
            axis.imshow(pixels)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{name}: {row['record_id']} ({row['flank_wear']:.0f} µm)", fontsize=8, color=INK)
    fig.suptitle("Tool photos at the model input size (384 x 128)", x=0.01, ha="left", color=INK, fontsize=11)
    save(fig, "12_image_examples.png")
    return written


# ----------------------------------------------------------------------------- main
def run_eda(
    data_dir: Path,
    out: Path,
    *,
    fs: float = 1000.0,
    pool: int = 1000,
    input_len: int = 96,
    horizon: int = 96,
    stride: int = 24,
    difficulty_pools: Sequence[int] = (1, 10, 100, 300, 1000),
    windows_per_tool: int = 5000,
    figures: bool = True,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    raw_rows, examples = raw_record_table(data_dir, fs)
    image_statistics(raw_rows, data_dir)
    retained_ids = {record.record_id for record in load_nonastreda(data_dir, pool=pool)}
    duplicates = duplicate_analysis(raw_rows, retained_ids)
    retained = [row for row in raw_rows if row["retained_after_dedup"]]
    raw_dynamics, raw_runs = label_dynamics(raw_rows, "raw")
    retained_dynamics, retained_runs = label_dynamics(retained, "retained")

    totals, tool_rows, fold_rows, timelines = protocol_windows(data_dir, pool, input_len, horizon, stride)
    acf, acf_summary = acf_rows(timelines[0], raw_rows, max_lag=24)
    difficulty = difficulty_by_pool(data_dir, difficulty_pools, input_len, windows_per_tool)

    class_counts = {split: Counter(row["image_label"] for row in rows) for split, rows in (("raw", raw_rows), ("retained", retained))}
    summary: dict[str, Any] = {
        "data_dir": str(data_dir.resolve()),
        "sampling_rate_hz_assumed": fs,
        "inventory": {
            "raw_records": len(raw_rows),
            "tools": len({row["tool"] for row in raw_rows}),
            "runs": raw_dynamics["runs"],
            "images": len(raw_rows),
            "records_per_tool": dict(Counter(row["tool"] for row in raw_rows)),
            "blocks_per_run": dict(Counter(Counter((r["tool"], r["run"]) for r in raw_rows).values())),
            "max_run_number": max(row["run"] for row in raw_rows),
            "max_block_number": max(row["block"] for row in raw_rows),
            "image_sizes": dict(Counter(f"{r['image_width']}x{r['image_height']}" for r in raw_rows)),
            "non_finite_force_values": sum(row["non_finite_values"] for row in raw_rows),
            "mat_vs_csv_label_mismatches": sum(row["mat_label"] != row["image_label"] for row in raw_rows),
        },
        "labels": {
            "class_counts": {split: {name: counts.get(name, 0) for name in CLASS_NAMES} for split, counts in class_counts.items()},
            "image_label_vs_tool_label": {
                f"{a}->{b}": count
                for (a, b), count in sorted(Counter((r["image_label"], r["tool_label"]) for r in retained).items())
            },
            "flank_wear_by_class": {
                name: _quantiles([r["flank_wear"] for r in retained if r["image_label"] == name]) for name in CLASS_NAMES
            },
            "spearman_flank_wear_vs_class": _spearman([r["flank_wear"] for r in retained], [CLASS_NAMES.index(r["image_label"]) for r in retained]),
            "dynamics_raw": raw_dynamics,
            "dynamics_retained": retained_dynamics,
        },
        "duplicates": duplicates,
        "force": {
            "duration_s": _quantiles([row["duration_s"] for row in raw_rows]),
            "samples_total_raw": sum(row["samples"] for row in raw_rows),
            "channel_mean": {c: _quantiles([row[f"{c}_mean"] for row in retained]) for c in CHANNELS},
            "channel_std": {c: _quantiles([row[f"{c}_std"] for row in retained]) for c in CHANNELS},
            "magnitude_sum_per_pass": _quantiles([row["magnitude_sum"] for row in retained]),
            "spearman_flank_wear_vs_mean_force": _spearman([r["flank_wear"] for r in retained], [r["magnitude_mean"] for r in retained]),
        },
        "trimming": {
            "rule": {
                "envelope_window_samples": PASS_TRIM_WINDOW_SAMPLES,
                "reference_fraction": PASS_TRIM_REFERENCE_FRACTION,
                "threshold_fraction": PASS_TRIM_THRESHOLD_FRACTION,
            },
            "records_trimmed": sum(r["trimmed_lead_samples"] + r["trimmed_runout_samples"] > 0 for r in raw_rows),
            "removed_fraction_of_samples": sum(r["trimmed_lead_samples"] + r["trimmed_runout_samples"] for r in raw_rows) / sum(r["samples"] for r in raw_rows),
            "records_with_idle_reference": sum(r["reference_is_idle"] for r in raw_rows),
            "records_with_residual_idle_lead_over_1s": sum(r["residual_lead_idle_samples"] > fs for r in raw_rows),
            "records_with_residual_idle_runout_over_1s": sum(r["residual_runout_idle_samples"] > fs for r in raw_rows),
            "residual_idle_lead_seconds": _quantiles([r["residual_lead_idle_samples"] / fs for r in raw_rows]),
            "retained_idle_fraction": _quantiles([r["retained_idle_fraction"] for r in raw_rows]),
            "retained_records_with_idle_reference": sum(r["reference_is_idle"] for r in retained),
        },
        "protocol_windows": totals,
        "persistence_and_alignment": {
            "eval_windows_target_class_change": float(np.mean([r["eval_windows_target_class_change"] for r in tool_rows if r["eval_windows_target_class_change"] is not None])),
            "eval_windows_input_spans_records": float(np.mean([r["eval_windows_input_spans_records"] for r in tool_rows if r["eval_windows_input_spans_records"] is not None])),
            "persistence_accuracy_lag0": float(np.mean([r["persistence_accuracy_lag0"] for r in tool_rows if r["persistence_accuracy_lag0"] is not None])),
            "persistence_accuracy_lag1": float(np.mean([r["persistence_accuracy_lag1"] for r in tool_rows if r["persistence_accuracy_lag1"] is not None])),
        },
        "autocorrelation": acf_summary,
        "normalization_shift": {
            "abs_test_z_mean": {c: _quantiles([abs(r[f"test_z_mean_{c}"]) for r in fold_rows]) for c in CHANNELS},
            "test_z_std": {c: _quantiles([r[f"test_z_std_{c}"] for r in fold_rows]) for c in CHANNELS},
        },
        "images": {
            "mean_luminance_by_class": {name: _quantiles([r["image_mean_luminance"] for r in retained if r["image_label"] == name]) for name in CLASS_NAMES},
            "edge_strength_by_class": {name: _quantiles([r["image_edge_strength"] for r in retained if r["image_label"] == name]) for name in CLASS_NAMES},
            "file_megabytes": _quantiles([r["image_bytes"] / 1e6 for r in raw_rows]),
            # Capture resolution differs by tool; resizing to 384x128 removes most of it.
            "full_resolution_images_by_tool": {
                tool: sum(r["tool"] == tool and r["image_width"] > 2000 for r in raw_rows)
                for tool in sorted({r["tool"] for r in raw_rows}, key=lambda t: int(t[1:]))
            },
            "full_resolution_images_by_class_retained": {
                name: sum(r["image_label"] == name and r["image_width"] > 2000 for r in retained)
                for name in CLASS_NAMES
            },
        },
    }
    for row in raw_rows:
        row.pop("_force_hash", None)
    _write_csv(out / "records.csv", raw_rows)
    _write_csv(out / "runs.csv", raw_runs + retained_runs)
    _write_csv(out / "tools.csv", tool_rows)
    _write_csv(out / "folds.csv", fold_rows)
    _write_csv(out / "difficulty_by_time_step.csv", difficulty)
    _write_csv(out / "autocorrelation.csv", acf)
    _write_csv(out / "trim_flags.csv", [
        {key: row[key] for key in ("record_id", "samples", "trimmed_lead_samples", "trimmed_runout_samples", "reference_idle_fraction",
                                    "reference_is_idle", "residual_lead_idle_samples", "residual_runout_idle_samples", "retained_idle_fraction",
                                    "alternative_trim_lead_samples", "alternative_trim_runout_samples", "retained_after_dedup")}
        for row in raw_rows
        if row["reference_is_idle"] or row["residual_lead_idle_samples"] > fs or row["residual_runout_idle_samples"] > fs
    ])
    if figures:
        summary["figures"] = make_figures(out, raw_rows, examples, tool_rows, fold_rows, difficulty, acf, timelines, data_dir, fs)
    summary["tables"] = sorted(path.name for path in out.glob("*.csv"))
    with (out / "eda_summary.json").open("w", encoding="utf-8") as file:
        json.dump(_jsonable(summary), file, indent=2)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("Nonastreda_Data"))
    parser.add_argument("--out", type=Path, default=Path("experiment_results/eda"))
    parser.add_argument("--fs", type=float, default=1000.0, help="force sampling rate in Hz")
    parser.add_argument("--pool", type=int, default=1000)
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--difficulty-pools", type=int, nargs="+", default=[1, 10, 100, 300, 1000])
    parser.add_argument("--windows-per-tool", type=int, default=5000)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    summary = run_eda(
        args.data_dir, args.out, fs=args.fs, pool=args.pool, input_len=args.input_len, horizon=args.horizon,
        stride=args.stride, difficulty_pools=args.difficulty_pools, windows_per_tool=args.windows_per_tool,
        figures=not args.no_figures,
    )
    print(json.dumps(_jsonable({key: summary[key] for key in ("inventory", "duplicates", "trimming", "persistence_and_alignment")}), indent=2))


if __name__ == "__main__":
    main()
