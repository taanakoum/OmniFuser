"""Optional image inputs: force spectrograms and a frozen pretrained photo encoder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from test_alignment_patch import make_nonastreda

from omnifuser.cross_validation import _load_finished_fold
from omnifuser.data import (
    NO_IMAGE,
    ForceNormalizer,
    WindowDataset,
    load_nonastreda,
    make_tool_timelines,
    pool_records,
)
from omnifuser.spectrogram import SpectrogramBank, SpectrogramSettings, pass_spectrogram
from omnifuser.train import train_fold

SMALL = SpectrogramSettings(window=64, hop=16)
TINY = {"num_rtd_layers": 2, "num_gat_layers": 1, "image_height": 32, "image_width": 96}


def test_spectrogram_rows_are_frequencies_and_images_are_standardized() -> None:
    samples = np.arange(4096)
    # Fx: a sine at bin 8 of a 64-sample window; Fy and Fz: noise
    force = np.stack(
        [np.sin(2 * np.pi * 8 * samples / 64), *np.random.default_rng(0).normal(size=(2, 4096))],
        axis=1,
    ).astype(np.float32)
    image = pass_spectrogram(force + 100.0, SMALL, height=33, width=40)
    assert image.shape == (3, 33, 40) and image.dtype == np.float32
    # 33 rows = the 33 STFT bins, so row 8 is the sine's frequency
    assert int(image[0].mean(axis=1).argmax()) == 8
    assert np.allclose(image.mean(axis=(1, 2)), 0.0, atol=1e-4)
    assert np.allclose(image.std(axis=(1, 2)), 1.0, atol=1e-2)
    with pytest.raises(ValueError, match="shorter than the spectrogram window"):
        pass_spectrogram(force[:32], SMALL, 33, 40)
    with pytest.raises(ValueError):
        SpectrogramSettings(window=64, hop=128)


def test_spectrogram_bank_keys_by_photo_path_and_blanks_missing_images(tmp_path: Path) -> None:
    root = make_nonastreda(tmp_path / "data", tools=(1,), runs=(1,), length=300)
    records = load_nonastreda(root, trim_pass=False)
    bank = SpectrogramBank(records, SMALL, 32, 96)
    assert len(bank) == len({record.image_path for record in records})
    assert tuple(bank.tensor(records[0].image_path).shape) == (3, 32, 96)
    assert torch.count_nonzero(bank.tensor(NO_IMAGE)) == 0
    # pooling after loading equals pooling in the loader, which spectrogram runs rely on
    pooled = load_nonastreda(root, trim_pass=False, pool=4)
    repooled = pool_records(records, 4)
    assert all(np.array_equal(a.force, b.force) for a, b in zip(pooled, repooled))


def test_lag_one_images_never_cover_the_forecast_window(tmp_path: Path) -> None:
    """Every image a window sees comes from a pass that ended before the window's input end."""
    root = make_nonastreda(tmp_path / "data", tools=(1, 2), runs=(1, 2), length=40)
    records = load_nonastreda(root, trim_pass=False)
    timelines = make_tool_timelines(records, image_lag=1)
    for timeline in timelines:
        owned = [r for r in records if r.tool == timeline.tool]
        owned.sort(key=lambda r: (r.run, r.block))
        pass_end = {}
        position = 0
        for record in owned:
            position += len(record.force)
            pass_end[record.image_path] = position
        data = WindowDataset([timeline], ForceNormalizer.fit([timeline]), 24, 12, 4)
        for index in range(len(data)):
            input_end = index * 4 + 24
            for path in data[index]["image_paths"]:
                assert path == NO_IMAGE or pass_end[path] <= input_end


def test_spectrogram_runs_need_lag_one_and_never_reuse_photo_folds(tmp_path: Path) -> None:
    root = make_nonastreda(tmp_path / "data", tools=(1, 2, 3), runs=(1, 2), length=200)
    common = {
        "data_dir": root, "test_tool": "T1", "epochs": 1, "batch_size": 2, "input_len": 24, "horizon": 12,
        "stride": 12, "device_name": "cpu", "trim_pass": False, "image_encoder": "compact",
        "max_train_windows": 4, "num_workers": 0, "config_overrides": TINY,
    }
    with pytest.raises(ValueError, match="image-lag 1"):
        train_fold(output_dir=tmp_path / "lag0", image_lag=0, spectrogram=SMALL, **common)

    out = tmp_path / "spectrogram"
    result = train_fold(output_dir=out, image_lag=1, spectrogram=SMALL, **common)
    assert result["spectrogram"] == {"window": 64, "hop": 16}
    assert np.isfinite(result["test"]["loss"])
    config = result["config"]
    # a photo run (no spectrogram) must not reuse this fold; the same spectrogram run may
    assert _load_finished_fold(out, "T1", config, {}, 1, 2) is None
    assert _load_finished_fold(out, "T1", config, {}, 1, 2, {"window": 64, "hop": 16}) == result


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
        num_gat_layers=1, pretrained=True, freeze_image_encoder=True,
    )
    model = OmniFuser(config)
    encoder = model.image_branch.encoder
    assert not any(p.requires_grad for p in encoder.features.parameters())
    assert all(p.requires_grad for p in encoder.projection.parameters())
    model.train()
    norms = [m for m in encoder.features.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert norms and not any(m.training for m in norms)
