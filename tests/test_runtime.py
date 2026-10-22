"""Runtime safeguards: CPU thread choice, the Windows WMI guard, cached frozen features."""
from __future__ import annotations

import platform
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from test_alignment_patch import make_nonastreda

import omnifuser
from omnifuser import resources
from omnifuser.config import ModelConfig
from omnifuser.data import NO_IMAGE, ImageBank, collate_windows
from omnifuser.models import OmniFuser
from omnifuser.train import FrozenFeatureBank, train_fold


def test_cpu_threads_respect_override_allocation_and_idle_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OMNIFUSER_CPU_THREADS", "OMP_NUM_THREADS", "SLURM_JOB_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(resources, "available_cpus", lambda: 8)
    monkeypatch.setattr(resources, "idle_cpus", lambda: 2.7)
    assert resources.cpu_threads(limit=4) == 2  # other programs hold the rest
    monkeypatch.setattr(resources, "idle_cpus", lambda: 0.3)
    assert resources.cpu_threads(limit=4) == 1  # never zero
    monkeypatch.setattr(resources, "idle_cpus", lambda: None)
    assert resources.cpu_threads(limit=4) == 4  # unknown load: the limit
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert resources.cpu_threads(limit=4) == 3
    monkeypatch.setenv("SLURM_JOB_ID", "1")  # a job's cores are its own: load is ignored
    monkeypatch.setattr(resources, "idle_cpus", lambda: 0.0)
    assert resources.cpu_threads(limit=4) == 3
    monkeypatch.setenv("OMNIFUSER_CPU_THREADS", "6")
    assert resources.cpu_threads(limit=4) == 6


def test_idle_cpu_measurement_is_bounded() -> None:
    idle = resources.idle_cpus(sample_seconds=0.05)
    assert idle is None or 0.0 <= idle <= (resources.os.cpu_count() or 1)


def test_wmi_guard_switches_platform_to_its_fallback_when_wmi_hangs(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(platform, "uname", lambda: release.wait())  # a WMI call that never returns
    monkeypatch.setattr(platform, "_wmi_query", lambda *a: ("10.0",), raising=False)
    try:
        assert omnifuser._guard_windows_wmi(timeout=0.2) is True
        with pytest.raises(OSError):
            platform._wmi_query("OS", "Version")  # pyright: ignore[reportAttributeAccessIssue]
    finally:
        release.set()
    monkeypatch.setattr(platform, "uname", lambda: None)  # answers at once
    monkeypatch.setattr(platform, "_wmi_query", lambda *a: ("10.0",), raising=False)
    assert omnifuser._guard_windows_wmi(timeout=0.2) is False


def _random_efficientnet(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchvision.models as tvm

    real = tvm.efficientnet_b0
    monkeypatch.setattr(tvm, "efficientnet_b0", lambda weights=None: real(weights=None))


def test_frozen_backbone_is_deterministic_and_cached_features_match_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _random_efficientnet(monkeypatch)
    torch.manual_seed(0)
    config = ModelConfig(
        input_len=12, horizon=6, image_height=32, image_width=96, num_rtd_layers=2,
        num_gat_layers=1, pretrained=True, freeze_image_encoder=True,
    )
    model = OmniFuser(config)
    encoder = model.image_branch.encoder
    model.train()
    pixels = torch.randn(4, 3, 32, 96)
    # in training mode a frozen backbone still runs without stochastic depth
    assert torch.equal(encoder.backbone_features(pixels), encoder.backbone_features(pixels))

    root = make_nonastreda(tmp_path / "data", tools=(1,), runs=(1,), length=40)
    paths = sorted((root / "tool").glob("*.jpg"))
    pixel_bank = ImageBank([*paths, NO_IMAGE], 32, 96)
    feature_bank = FrozenFeatureBank(pixel_bank, encoder, torch.device("cpu"))
    assert len(feature_bank) == len(paths) + 1
    samples = [
        {
            "force": np.zeros((12, 3), np.float32), "force_target": np.zeros((6, 3), np.float32),
            "class_target": np.zeros(6, np.int64), "shown_label": np.int64(0),
            "image_paths": tuple([paths[0]] * 6 + [paths[1]] * 6),
            "cycle": np.ones(12, np.int64), "batch": np.ones(12, np.int64), "relative": np.zeros(12, np.int64),
        },
        {
            "force": np.ones((12, 3), np.float32), "force_target": np.zeros((6, 3), np.float32),
            "class_target": np.zeros(6, np.int64), "shown_label": np.int64(0),
            "image_paths": tuple([NO_IMAGE] * 12),
            "cycle": np.ones(12, np.int64), "batch": np.ones(12, np.int64), "relative": np.zeros(12, np.int64),
        },
    ]
    with_pixels = collate_windows(samples, 32, 96, pixel_bank)
    with_features = collate_windows(samples, 32, 96, feature_bank)
    assert with_features["images"].shape == (2, 2, encoder.feature_width)
    model.eval()
    with torch.no_grad():
        arguments = [with_pixels[key] for key in ("force", "images", "image_indices", "cycle", "batch", "relative")]
        reference = model(*arguments)
        arguments[1] = with_features["images"]
        cached = model(*arguments)
    assert torch.allclose(reference["force"], cached["force"], atol=1e-5)
    assert torch.allclose(reference["class_logits"], cached["class_logits"], atol=1e-5)


def test_frozen_photo_runs_train_on_cached_features(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _random_efficientnet(monkeypatch)
    root = make_nonastreda(tmp_path / "data", tools=(1, 2, 3), runs=(1,), length=120)
    result = train_fold(
        data_dir=root, test_tool="T1", output_dir=tmp_path / "out", epochs=1, batch_size=2,
        input_len=24, horizon=12, stride=12, device_name="cpu", trim_pass=False, pool=4,
        max_train_windows=4, max_eval_windows=2, num_workers=0,
        config_overrides={
            "num_rtd_layers": 2, "num_gat_layers": 1, "image_height": 32, "image_width": 96,
            "pretrained": True, "freeze_image_encoder": True,
        },
    )
    assert result["training"]["cached_image_features"] is True
    assert np.isfinite(result["test"]["loss"])
