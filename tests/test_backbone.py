"""OmniFuser as a backbone (v1.1): ``encode()`` and several photo views per step."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from omnifuser.config import ModelConfig
from omnifuser.models import OmniFuser


def _config(**changes: object) -> ModelConfig:
    base = ModelConfig(
        d_model=16, input_len=12, horizon=4, num_rtd_layers=2, rtd_rank=4,
        num_gat_layers=1, num_gat_heads=4, num_fusion_layers=2, num_landmarks=2,
        image_encoder="compact", image_height=32, image_width=64, dropout=0.0,
    )
    return replace(base, **changes)


def _features(width: int, batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(force, precomputed photo features, per-step photo indices)."""
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(batch, 12, 3, generator=generator),
        torch.randn(batch, 3, width, generator=generator),
        torch.randint(0, 3, (batch, 12), generator=generator),
    )


def test_encode_returns_the_fused_sequence_that_the_heads_read() -> None:
    torch.manual_seed(0)
    model = OmniFuser(_config()).eval()
    inputs = _features(96)
    with torch.no_grad():
        fused = model.encode(*inputs)
        outputs = model(*inputs)
    assert fused.shape == (2, 12, 16)
    expected = model.direct_force(fused[:, -1]).view(2, 4, 3)
    assert torch.equal(outputs["force"], expected)
    assert torch.equal(outputs["class_logits"], model.direct_class(fused[:, -1]).view(2, 4, 3))


def test_encode_applies_modality_masking() -> None:
    torch.manual_seed(0)
    model = OmniFuser(_config()).eval()
    force, images, indices = _features(96)
    with torch.no_grad():
        assert not torch.allclose(model.encode(force, images, indices), model.encode(force, images + 1.0, indices))
        assert torch.equal(model.encode(force, images, indices, drop_modality="image"),
                           model.encode(force, images + 1.0, indices, drop_modality="image"))


def test_several_views_widen_the_projection_and_need_precomputed_features() -> None:
    torch.manual_seed(0)
    one, two = OmniFuser(_config()), OmniFuser(_config(image_views=2))
    projection = two.image_branch.encoder.projection
    assert projection.in_features == 2 * 96
    extra = sum(p.numel() for p in two.parameters()) - sum(p.numel() for p in one.parameters())
    assert extra == 96 * 16  # only the projection's extra input columns
    with torch.no_grad():
        assert two.eval().encode(*_features(2 * 96)).shape == (2, 12, 16)
        with pytest.raises(ValueError, match="2 view"):
            two.encode(*_features(96))
        force, _, indices = _features(96)
        with pytest.raises(ValueError, match="precomputed features only"):
            two.encode(force, torch.randn(2, 3, 3, 32, 64), indices)
    with pytest.raises(ValueError, match="image_views"):
        _config(image_views=0)


def test_one_view_keeps_the_protocol_model_unchanged() -> None:
    model = OmniFuser(ModelConfig(), load_pretrained=False)
    assert sum(p.numel() for p in model.parameters()) == 4_183_600


def test_omnifuser_training_refuses_several_views(tmp_path: Path) -> None:
    from omnifuser.train import train_fold

    with pytest.raises(ValueError, match="one photo per step"):
        train_fold(data_dir=tmp_path, test_tool="T1", output_dir=tmp_path / "run",
                   config_overrides={"image_views": 2})
