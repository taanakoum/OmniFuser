"""Every number and figure quoted in END_TO_END_SIM.md, measured on the real data.

Run from the repository root after any substantial change to the data pipeline, the
model or the training loop, then update END_TO_END_SIM.md from its output:

    python scripts/end_to_end_trace.py --data-dir Nonastreda_Data

It prints the numbers as JSON (also written to --json) and redraws the figures in
docs/end_to_end/. It needs about 2 GB of free memory (the unpooled force file) and the
cached ImageNet weights of EfficientNet-B0 (TRAINING_RUNBOOK.md step 0). The model is
untrained (seed 42) and in eval mode, so the forward pass is deterministic; its outputs
illustrate shapes and scales, not results.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import omnifuser  # noqa: F401  # before torch: guards against a stuck Windows WMI service

# isort: split
import numpy as np
import torch
from PIL import Image

from omnifuser.config import ModelConfig
from omnifuser.data import (
    IMAGE_MEAN,
    IMAGE_STD,
    PASS_TRIM_REFERENCE_FRACTION,
    PASS_TRIM_THRESHOLD_FRACTION,
    PASS_TRIM_WINDOW_SAMPLES,
    ForceNormalizer,
    ImageBank,
    WindowDataset,
    _nonastreda_span_bounds,
    collate_windows,
    leave_one_tool_out_folds,
    load_nonastreda,
    make_tool_timelines,
    pool_records,
)
from omnifuser.models import OmniFuser, multitask_loss
from omnifuser.resources import cpu_threads
from omnifuser.spectrogram import (
    ForceWindowDataset,
    SpectrogramSettings,
    raw_passes_by_tool,
)
from omnifuser.train import FrozenFeatureBank, _accumulation_group_sample_count

POOL, INPUT_LEN, HORIZON, TRAIN_STRIDE = 1000, 96, 96, 24
EVAL_STRIDE = max(TRAIN_STRIDE, HORIZON)
EXAMPLE_TOOL, EXAMPLE_PASS = "T10", "T10R1B1"
# Reference palette (dataviz skill): surface, ink, blue, orange; sequential blue ramp.
SURFACE, INK, MUTED, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def rounded(value: Any, digits: int = 4) -> Any:
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item, digits) for item in value]
    if isinstance(value, (np.floating, float)):
        return round(float(value), digits)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return rounded(value.tolist(), digits)
    if isinstance(value, torch.Tensor):
        return rounded(value.detach().cpu().numpy(), digits)
    return value


def stats(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float()
    return {"mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()), "max": float(values.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Numbers and figures for END_TO_END_SIM.md.")
    parser.add_argument("--data-dir", type=Path, default=Path("Nonastreda_Data"))
    parser.add_argument("--json", type=Path, default=Path("experiment_results/end_to_end/trace.json"))
    parser.add_argument("--figures", type=Path, default=Path("docs/end_to_end"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(cpu_threads(limit=4))
    out: dict[str, Any] = {}

    # 1. Raw files ---------------------------------------------------------------------
    labels = (args.data_dir / "labels.csv").read_text(encoding="utf-8-sig").strip().splitlines()
    photos = sorted((args.data_dir / "tool").glob("*.jpg"))
    with Image.open(photos[0]) as photo:
        photo_size = photo.size
    out["raw_files"] = {"label_rows": len(labels) - 1, "photos": len(photos), "photo_size_wh": photo_size,
                        "label_header": labels[0], "label_example": labels[1]}

    # 2. One pass, untrimmed: the trimming rule step by step ---------------------------
    untrimmed = load_nonastreda(args.data_dir, pool=1, trim_pass=False)
    record = next(r for r in untrimmed if r.record_id == EXAMPLE_PASS)
    force = record.force
    start, end = _nonastreda_span_bounds(force)
    magnitude = np.square(force.astype(np.float64)).sum(axis=1)
    from scipy.ndimage import uniform_filter1d

    envelope = np.sqrt(uniform_filter1d(magnitude, size=PASS_TRIM_WINDOW_SAMPLES, mode="nearest"))
    reference_length = max(PASS_TRIM_WINDOW_SAMPLES, int(len(force) * PASS_TRIM_REFERENCE_FRACTION))
    reference_level = float(np.median(envelope[:reference_length]))
    trimmed = force[start:end]
    steps = len(trimmed) // POOL
    pooled = trimmed[: steps * POOL].reshape(steps, POOL, 3).mean(axis=1)
    out["example_pass"] = {
        "record_id": EXAMPLE_PASS, "raw_samples": len(force), "first_raw_sample_N": force[0],
        "reference_length": reference_length, "reference_level_N": reference_level,
        "threshold_N": reference_level * PASS_TRIM_THRESHOLD_FRACTION, "trim_start": start, "trim_end": end,
        "trimmed_lead": start, "trimmed_runout": len(force) - end, "kept_samples": len(trimmed),
        "pool_dropped": len(trimmed) % POOL, "pooled_steps": steps,
        "first_pooled_step_N": pooled[0], "first_1000_raw_mean_N": trimmed[:POOL].mean(axis=0),
        "label": record.label,
    }
    figure_pass = (force[:, 0].copy(), envelope.copy(), reference_level, start, end, pooled[:, 0].copy())
    out["untrimmed_total_samples"] = int(sum(len(r.force) for r in untrimmed))
    del untrimmed, record, force, magnitude, envelope

    # 3. The protocol's data: trimmed, deduplicated, then pooled -----------------------
    raw = load_nonastreda(args.data_dir, pool=1)
    all_passes = load_nonastreda(args.data_dir, pool=POOL, dedup_identical_forces=False)
    pooled_records = pool_records(raw, POOL)
    lengths = np.array([len(r.force) for r in pooled_records])
    out["dataset"] = {
        "passes_dedup": len(raw), "passes_all": len(all_passes),
        "classes_dedup": np.bincount([r.label for r in raw], minlength=3),
        "classes_all": np.bincount([r.label for r in all_passes], minlength=3),
        "pooled_steps_min_median_max": [int(lengths.min()), float(np.median(lengths)), int(lengths.max())],
        "raw_force_GB": sum(r.force.nbytes for r in raw) / 2**30,
        "trimmed_lead_total": sum(r.trimmed_lead_samples for r in raw),
        "trimmed_runout_total": sum(r.trimmed_runout_samples for r in raw),
    }
    del all_passes

    # 4. Timelines, folds, normalizer, windows -----------------------------------------
    timelines = {t.tool: t for t in make_tool_timelines(pooled_records, image_lag=0)}
    lag1 = {t.tool: t for t in make_tool_timelines(pooled_records, image_lag=1)}
    example = timelines[EXAMPLE_TOOL]
    tool_records = sorted((r for r in pooled_records if r.tool == EXAMPLE_TOOL), key=lambda r: (r.run, r.block))
    boundaries = np.cumsum([len(r.force) for r in tool_records])
    out["timeline"] = {
        "steps_per_tool": {tool: len(t.force) for tool, t in timelines.items()},
        "total_steps": sum(len(t.force) for t in timelines.values()),
        "example_tool": EXAMPLE_TOOL, "example_passes": len(tool_records),
        "example_first_pass_ends": boundaries[:4],
        "example_pass_ids": [r.record_id for r in tool_records[:4]],
        "example_codes_at_steps_0_118_119": {
            "image_code": example.image_codes[[0, 118, 119]], "label": example.labels[[0, 118, 119]],
            "cycle": example.cycle_codes[[0, 118, 119]], "batch": example.batch_codes[[0, 118, 119]],
            "relative": example.relative_codes[[0, 7, 8, 118, 119]],
        },
        "lag1_first_image_paths": [str(p) for p in lag1[EXAMPLE_TOOL].image_paths[:2]],
        "lag1_image_labels_at_0_119": lag1[EXAMPLE_TOOL].image_labels[[0, 119]],
        "windows_per_tool_stride24": {
            tool: max((len(t.force) - INPUT_LEN - HORIZON) // TRAIN_STRIDE + 1, 0) for tool, t in timelines.items()
        },
    }
    out["timeline"]["total_windows_stride24"] = sum(out["timeline"]["windows_per_tool_stride24"].values())
    folds = list(leave_one_tool_out_folds(list(timelines)))
    out["folds"] = [{"test": f.test_tools[0], "validation": f.validation_tools[0], "train": list(f.train_tools)} for f in folds]
    train_counts = {}
    for fold in folds:
        train_counts[fold.test_tools[0]] = len(WindowDataset([timelines[t] for t in fold.train_tools],
                                                            ForceNormalizer(np.zeros(3, np.float32), np.ones(3, np.float32)),
                                                            INPUT_LEN, HORIZON, TRAIN_STRIDE))
    out["train_windows_per_fold"] = train_counts
    fold = next(f for f in folds if f.test_tools[0] == EXAMPLE_TOOL)
    normalizer = ForceNormalizer.fit([timelines[t] for t in fold.train_tools])
    train_data = WindowDataset([timelines[t] for t in fold.train_tools], normalizer, INPUT_LEN, HORIZON, TRAIN_STRIDE)
    validation_data = WindowDataset([timelines[t] for t in fold.validation_tools], normalizer, INPUT_LEN, HORIZON, EVAL_STRIDE)
    test_data = WindowDataset([example], normalizer, INPUT_LEN, HORIZON, EVAL_STRIDE)
    out["fold_T10"] = {
        "normalizer_mean_N": normalizer.mean, "normalizer_scale_N": normalizer.scale,
        "windows": {"train": len(train_data), "validation": len(validation_data), "test": len(test_data)},
        "train_batches_of_32": -(-len(train_data) // 32), "last_batch": len(train_data) % 32 or 32,
        "accumulation_2_group_sizes_first_last": [
            _accumulation_group_sample_count(len(train_data), 32, 1, 2),
            _accumulation_group_sample_count(len(train_data), 32, -(-len(train_data) // 32), 2),
        ],
    }

    # 5. One test window that crosses a pass boundary -----------------------------------
    item0, item1 = test_data[0], test_data[1]
    window_start = test_data.locate(1)[1]
    unique_paths = list(dict.fromkeys(item1["image_paths"]))
    raw_step = example.force[window_start]
    out["example_window"] = {
        "index": 1, "window_start": window_start, "input_steps": [window_start, window_start + INPUT_LEN - 1],
        "target_steps": [window_start + INPUT_LEN, window_start + INPUT_LEN + HORIZON - 1],
        "photos": [Path(p).stem for p in unique_paths],
        "steps_per_photo": [sum(Path(p) == Path(u) for p in item1["image_paths"]) for u in unique_paths],
        "raw_force_at_input_start_N": raw_step, "normalized_force_at_input_start": item1["force"][0],
        "class_targets_counts": np.bincount(item1["class_target"], minlength=3),
        "shown_label": int(item1["shown_label"]),
        "cycle_batch_relative_first_step": [int(item1["cycle"][0]), int(item1["batch"][0]), int(item1["relative"][0])],
        "window0_photos": len(set(item0["image_paths"])),
    }

    # 6. Photos: resize, normalization, frozen features ---------------------------------
    bank = ImageBank.from_timelines(list(timelines.values()), 128, 384)
    pixel = bank.pixels[bank.index[Path(unique_paths[0])]][0, 0]
    out["photo"] = {
        "cached_photos": len(bank), "cache_MB": bank.pixels.nbytes / 2**20, "pixel_00_rgb": pixel,
        "pixel_00_normalized": (pixel / 255.0 - IMAGE_MEAN) / IMAGE_STD,
    }
    torch.manual_seed(42)
    model = OmniFuser(ModelConfig(pretrained=True, freeze_image_encoder=True)).eval()
    started = time.perf_counter()
    features = FrozenFeatureBank(bank, model.image_branch.encoder, torch.device("cpu"))
    out["frozen_features"] = {
        "photos_plus_blank": len(features), "seconds_cpu": time.perf_counter() - started,
        "width": features.features.shape[1], "example_stats": stats(features.tensor(Path(unique_paths[0]))),
        "blank_image_stats": stats(features.tensor(Path("<no previous image>"))),
    }

    # 7. Forward pass through the model, block by block ---------------------------------
    batch = collate_windows([item0, item1], 128, 384, features)
    captured: dict[str, Any] = {}

    def keep(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output
        return hook

    model.temporal_embedding.register_forward_hook(keep("embedding"))
    model.temporal_encoder.register_forward_hook(keep("mrte"))
    model.image_branch.encoder.register_forward_hook(keep("image_encoded"))
    model.image_branch.register_forward_hook(keep("image_branch"))
    for index, layer in enumerate(model.fusion_layers):
        layer.register_forward_hook(keep(f"c2f_{index}"))
    rtd = []
    for block in model.temporal_encoder.layers:  # type: ignore[union-attr]
        rtd.append({"input_steps": block.input_steps, "pad": (-block.input_steps) % block.stride, "output_steps": block.output_steps})
    with torch.no_grad():
        outputs = model(batch["force"], batch["images"], batch["image_indices"], batch["cycle"], batch["batch"], batch["relative"])
        _loss, components = multitask_loss(outputs, batch["force_target"], batch["class_target"])
    layer0 = model.fusion_layers[0]
    out["forward"] = {
        "batch_shapes": {key: list(value.shape) for key, value in batch.items()},
        "image_indices_window1_first_last": [int(batch["image_indices"][1, 0]), int(batch["image_indices"][1, -1])],
        "embedding": stats(captured["embedding"]), "rtd_blocks": rtd, "mrte": stats(captured["mrte"]),
        "image_encoded_shape": list(captured["image_encoded"].shape), "image_branch": stats(captured["image_branch"]),
        "c2f_round1": {"fused": stats(captured["c2f_0"][0]), "alignment": float(captured["c2f_0"][1]),
                       "orthogonality": float(captured["c2f_0"][2])},
        "gate_mix_initial": float(torch.sigmoid(layer0.gate_mix_r.detach())),  # type: ignore[union-attr]
        "force_output": stats(outputs["force"]), "logits_output": stats(outputs["class_logits"]),
        "loss": {key: float(value) for key, value in components.items()},
        "ln3": float(np.log(3.0)),
        "parameters": {"total": sum(p.numel() for p in model.parameters()),
                       "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)},
    }

    # 8. Spectrogram variant for the same window ---------------------------------------
    settings = SpectrogramSettings()
    passes = raw_passes_by_tool(raw, POOL)
    spectrogram_data = ForceWindowDataset([example], normalizer, passes, POOL, settings, 128, 384, INPUT_LEN, HORIZON, EVAL_STRIDE)
    segments = spectrogram_data.segments(0, window_start)
    started = time.perf_counter()
    spectrogram_item = spectrogram_data[1]
    seconds = time.perf_counter() - started
    segment_samples = [(end_step - start_step) * POOL for _, start_step, end_step in segments]
    out["spectrogram"] = {
        "segments_pass_start_end": segments, "segment_samples": segment_samples,
        "stft_frames": [(n - settings.window) // settings.hop + 1 for n in segment_samples],
        "stft_bins": settings.window // 2 + 1, "images_shape": list(spectrogram_item["images"].shape),
        "image_index_first_last": [int(spectrogram_item["image_index"][0]), int(spectrogram_item["image_index"][-1])],
        "seconds_for_window": seconds,
        "full_window_frames": (INPUT_LEN * POOL - settings.window) // settings.hop + 1,
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    result = rounded(out)
    args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not args.no_figures:
        draw_figures(args.figures, figure_pass, example, boundaries, normalizer, window_start,
                     spectrogram_item["images"][-1], segments[-1])


def draw_figures(directory, figure_pass, example, boundaries, normalizer, window_start, maps, segment) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    })

    # Figure 1: trimming and pooling of one pass.
    fx, envelope, reference, start, end, pooled_fx = figure_pass
    seconds = np.arange(len(fx)) / 1000.0
    shown = slice(None, None, 10)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    top.plot(seconds[shown], envelope[shown], color=BLUE, linewidth=1)
    top.axhline(reference * PASS_TRIM_THRESHOLD_FRACTION, color=MUTED, linestyle="--", linewidth=1)
    top.text(seconds[len(fx) // 2], reference * PASS_TRIM_THRESHOLD_FRACTION, "threshold = 0.5 x reference level",
             va="bottom", ha="center", color=MUTED)
    for axis in (top, bottom):
        axis.axvspan(0, start / 1000, color="#e6e5e0", zorder=0)
        axis.axvspan(end / 1000, seconds[-1], color="#e6e5e0", zorder=0)
    top.set_ylabel("envelope |F| (N)")
    top.set_title(f"{EXAMPLE_PASS}: trimming keeps samples {start:,}-{end:,} (grey = removed)", loc="left")
    bottom.plot(seconds[shown], fx[shown], color=BLUE, linewidth=0.6, alpha=0.6)
    pooled_time = start / 1000 + np.arange(len(pooled_fx)) + 0.5
    bottom.plot(pooled_time, pooled_fx, color=ORANGE, linewidth=2)
    bottom.text(pooled_time[-1], pooled_fx[-1], "  1 s means (pool 1000)", color=INK, va="center")
    bottom.text(seconds[len(fx) // 3], fx.min() + 5, "raw Fx, 1 kHz (every 10th sample drawn)", color=INK, va="bottom")
    bottom.set_xlabel("time in the pass (s)")
    bottom.set_ylabel("Fx (N)")
    fig.tight_layout()
    fig.savefig(directory / "01_pass_trimming_pooling.png", dpi=130)
    plt.close(fig)

    # Figure 2: the tool timeline and one evaluation window.
    shown_steps = 400
    fx_norm = (example.force[:shown_steps, 0] - normalizer.mean[0]) / normalizer.scale[0]
    fig, axis = plt.subplots(figsize=(10, 3.6))
    axis.plot(np.arange(shown_steps), fx_norm, color=BLUE, linewidth=2)
    axis.axvspan(window_start, window_start + INPUT_LEN, color="#cde2fb", zorder=0)
    axis.axvspan(window_start + INPUT_LEN, window_start + INPUT_LEN + HORIZON, color="#fbd9ca", zorder=0)
    top = fx_norm.max()
    axis.text(window_start + 2, top, "input: 96 steps", va="top", color=INK)
    axis.text(window_start + INPUT_LEN + 2, top, "forecast target: 96 steps", va="top", color=INK)
    for position, boundary in enumerate(boundaries):
        if boundary >= shown_steps:
            break
        axis.axvline(boundary, color=MUTED, linewidth=1, linestyle=":")
        axis.text(boundary + 1, fx_norm.min(), f"pass {position + 2}", color=MUTED, va="bottom")
    axis.set_xlim(0, shown_steps)
    axis.set_xlabel(f"{EXAMPLE_TOOL} timeline step (1 step = 1 s)")
    axis.set_ylabel("Fx, z-scored")
    axis.set_title(f"{EXAMPLE_TOOL} test window 1 (start {window_start}); dotted lines = pass boundaries", loc="left")
    fig.tight_layout()
    fig.savefig(directory / "02_timeline_window.png", dpi=130)
    plt.close(fig)

    # Figure 3: the 7 spectrogram maps of the window's second pass segment.
    ramp = LinearSegmentedColormap.from_list("ramp", RAMP)
    names = ["STFT Fx", "STFT Fy", "STFT Fz", "FFT Fx", "FFT Fy", "FFT Fz", "recurrence"]
    fig, axes = plt.subplots(4, 2, figsize=(10, 9))
    for index, axis in enumerate(axes.flat):
        axis.grid(False)
        if index >= len(names):
            axis.axis("off")
            continue
        axis.imshow(maps[index].numpy(), cmap=ramp, aspect="auto", origin="upper")
        axis.set_title(names[index], loc="left", fontsize=9)
        axis.set_xticks([0, 383])
        axis.set_xticklabels(["start", "end"])
        axis.set_yticks([0, 127])
        axis.set_yticklabels(["0 Hz", "500 Hz"] if index < 6 else ["start", "end"])
    _, start_step, end_step = segment
    fig.suptitle(f"Spectrogram maps of steps {start_step}-{end_step - 1} ({(end_step - start_step) * POOL:,} raw samples), "
                 "z-scored per map; darker = higher", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(directory / "03_spectrogram_maps.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
