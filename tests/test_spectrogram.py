"""Optional image inputs: force-window spectrograms and a frozen pretrained photo encoder."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from test_alignment_patch import make_nonastreda

from omnifuser.cross_validation import _load_finished_fold
from omnifuser.data import (
    ForceNormalizer,
    collate_windows,
    load_nonastreda,
    make_tool_timelines,
    pool_records,
)
from omnifuser.spectrogram import (
    IMAGE_CHANNELS,
    ForceWindowDataset,
    SpectrogramSettings,
    force_window_image,
    raw_passes_by_tool,
)
from omnifuser.train import model_config, train_fold

SMALL = SpectrogramSettings(window=64, hop=16)
TINY = {"num_rtd_layers": 2, "num_gat_layers": 1, "image_height": 32, "image_width": 96}


def test_window_image_maps_share_frequency_rows_and_are_standardized() -> None:
    samples = np.arange(4096)
    # Fx: a sine at bin 8 of a 64-sample window; Fy and Fz: noise
    force = np.stack(
        [np.sin(2 * np.pi * 8 * samples / 64), *np.random.default_rng(0).normal(size=(2, 4096))],
        axis=1,
    ).astype(np.float32)
    image = force_window_image(force + 100.0, SMALL, height=33, width=40)
    assert tuple(image.shape) == (IMAGE_CHANNELS, 33, 40) and image.dtype == torch.float32
    # 33 rows = the 33 STFT bins: row 8 is the sine's frequency in the local map (channel
    # 0) and in the global spectrum (channel 3), which is constant along time
    assert int(image[0].mean(dim=1).argmax()) == 8
    assert int(image[3, :, 0].argmax()) == 8
    assert torch.allclose(image[3], image[3, :, :1].expand(-1, 40))
    assert torch.allclose(image.mean(dim=(1, 2)), torch.zeros(IMAGE_CHANNELS), atol=1e-4)
    assert torch.allclose(image.std(dim=(1, 2)), torch.ones(IMAGE_CHANNELS), atol=1e-2)
    # a segment shorter than one STFT frame (one pooled step can be) still yields an image
    assert torch.isfinite(force_window_image(force[:20], SMALL, 33, 40)).all()
    with pytest.raises(ValueError):
        SpectrogramSettings(window=64, hop=128)
    with pytest.raises(ValueError):
        SpectrogramSettings(recurrence_rate=1.0)


def _force_windows(tmp_path: Path, pool: int = 4, input_len: int = 40) -> tuple[ForceWindowDataset, list]:
    root = make_nonastreda(tmp_path / "data", tools=(1,), runs=(1, 2), length=120)
    raw = load_nonastreda(root, trim_pass=False)
    timelines = make_tool_timelines(pool_records(raw, pool))
    data = ForceWindowDataset(
        timelines, ForceNormalizer.fit(timelines), raw_passes_by_tool(raw, pool), pool,
        SMALL, 32, 96, input_len=input_len, horizon=8, stride=8,
    )
    return data, raw


def test_window_images_use_only_the_input_samples_one_image_per_pass_segment(tmp_path: Path) -> None:
    pool = 4
    data, raw = _force_windows(tmp_path, pool)
    passes = data.passes[0]
    spans = 0
    for index in range(len(data)):
        _, window_start = data.locate(index)
        segments = data.segments(0, window_start)
        item = data[index]
        assert "image_paths" not in item
        assert tuple(item["images"].shape) == (len(segments), IMAGE_CHANNELS, 32, 96)
        # each step points at the image of the pass segment it belongs to
        for position, (_, start, end) in enumerate(segments):
            assert (item["image_index"][start - window_start : end - window_start] == position).all()
        assert sum(end - start for _, start, end in segments) == data.input_len
        spans += len(segments) > 1
    assert spans, "some windows must cross a pass boundary"

    # Changing every raw sample after a window's input end, or before its start, leaves
    # its images unchanged; changing one inside the input changes them.
    index = next(i for i in range(len(data)) if len(data.segments(0, data.locate(i)[1])) > 1)
    _, window_start = data.locate(index)
    input_end = window_start + data.input_len
    before = data[index]["images"].clone()
    for start, samples in zip(passes.starts, passes.raw):
        steps = np.arange(len(samples)) // pool + start
        outside = (steps >= input_end) | (steps < window_start)
        samples[outside] = np.random.default_rng(1).normal(0, 50, (int(outside.sum()), 3))
    assert torch.equal(data[index]["images"], before)
    pass_index, start, _ = data.segments(0, window_start)[0]
    passes.raw[pass_index][(start - passes.starts[pass_index]) * pool + 1] += 1000.0
    assert not torch.equal(data[index]["images"], before)
    assert len(raw) == len(passes.raw)


def test_window_images_collate_with_padding(tmp_path: Path) -> None:
    data, _ = _force_windows(tmp_path, input_len=16)  # shorter than a pass (~32 steps)
    counts = [len(data.segments(0, data.locate(i)[1])) for i in range(len(data))]
    fewest, most = counts.index(min(counts)), counts.index(max(counts))
    assert counts[fewest] < counts[most]
    batch = collate_windows([data[fewest], data[most]])
    assert tuple(batch["images"].shape) == (2, counts[most], IMAGE_CHANNELS, 32, 96)
    assert tuple(batch["image_indices"].shape) == (2, data.input_len)
    # the padded slots of the shorter row are never referenced
    assert int(batch["image_indices"][0].max()) == counts[fewest] - 1


def test_raw_passes_must_match_the_pooled_timeline(tmp_path: Path) -> None:
    root = make_nonastreda(tmp_path / "data", tools=(1,), runs=(1,), length=120)
    raw = load_nonastreda(root, trim_pass=False)
    timelines = make_tool_timelines(pool_records(raw, 4))
    with pytest.raises(ValueError, match="same load"):
        ForceWindowDataset(
            timelines, ForceNormalizer.fit(timelines), raw_passes_by_tool(raw, 5), 4,
            SMALL, 32, 96, input_len=40, horizon=8, stride=8,
        )
    # pooling after loading equals pooling in the loader, which spectrogram runs rely on
    pooled = load_nonastreda(root, trim_pass=False, pool=4)
    assert all(np.array_equal(a.force, b.force) for a, b in zip(pooled, pool_records(raw, 4)))


def test_spectrogram_runs_use_lag_zero_and_never_reuse_photo_folds(tmp_path: Path) -> None:
    root = make_nonastreda(tmp_path / "data", tools=(1, 2, 3), runs=(1, 2), length=200)
    common = {
        "data_dir": root, "test_tool": "T1", "epochs": 1, "batch_size": 2, "input_len": 24, "horizon": 12,
        "stride": 12, "device_name": "cpu", "trim_pass": False, "image_encoder": "compact", "pool": 4,
        "max_train_windows": 4, "num_workers": 0, "config_overrides": TINY,
    }
    with pytest.raises(ValueError, match="image-lag 0"):
        train_fold(output_dir=tmp_path / "lag1", image_lag=1, spectrogram=SMALL, **common)

    out = tmp_path / "spectrogram"
    result = train_fold(output_dir=out, image_lag=0, spectrogram=SMALL, **common)
    assert result["spectrogram"] == {"window": 64, "hop": 16, "recurrence_rate": 0.1}
    assert result["config"]["image_channels"] == IMAGE_CHANNELS
    assert np.isfinite(result["test"]["loss"])
    # cross-validation builds the expected config the same way, so it reuses this fold
    config = asdict(model_config(
        d_model=32, input_len=24, horizon=12, image_encoder="compact", prediction_head="direct",
        config_overrides=TINY, spectrogram=SMALL,
    ))
    assert config == result["config"]
    # a photo run (no spectrogram) must not reuse this fold; the same spectrogram run may
    assert _load_finished_fold(out, "T1", config, {}, 1, 2) is None
    spectrogram = {"window": 64, "hop": 16, "recurrence_rate": 0.1}
    assert _load_finished_fold(out, "T1", config, {}, 1, 2, spectrogram) == result


def test_frozen_pretrained_encoder_keeps_its_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchvision.models as tvm

    from omnifuser.config import ModelConfig
    from omnifuser.models import OmniFuser

    real = tvm.efficientnet_b0
    monkeypatch.setattr(tvm, "efficientnet_b0", lambda weights=None: real(weights=None))
    with pytest.raises(ValueError, match="needs pretrained"):
        ModelConfig(freeze_image_encoder=True)
    config = ModelConfig(
        input_len=24, horizon=12, image_height=32, image_width=96, num_rtd_layers=2,
        num_gat_layers=1, pretrained=True, freeze_image_encoder=True, image_channels=IMAGE_CHANNELS,
    )
    model = OmniFuser(config)
    encoder = model.image_branch.encoder
    assert not any(p.requires_grad for p in encoder.features.parameters())
    assert all(p.requires_grad for p in encoder.projection.parameters())
    # the spectrogram's image generator trains even when the backbone is frozen
    assert all(p.requires_grad for p in encoder.generator.parameters())
    model.train()
    norms = [m for m in encoder.features.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert norms and not any(m.training for m in norms)
    outputs = model(
        torch.randn(2, 24, 3), torch.randn(2, 1, IMAGE_CHANNELS, 32, 96), torch.zeros(2, 24, dtype=torch.long)
    )
    assert tuple(outputs["force"].shape) == (2, 12, 3)
    # photos keep the paper's encoder input: no generator
    photo = OmniFuser(ModelConfig(input_len=24, horizon=12, num_rtd_layers=2, num_gat_layers=1, image_encoder="compact"))
    assert isinstance(photo.image_branch.encoder.generator, torch.nn.Identity)
