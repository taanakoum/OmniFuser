from __future__ import annotations

from typing import Any

import pytest
import torch

from omnifuser.cli import ARCHITECTURE_OPTIONS, build_parser, protocol_kwargs
from omnifuser.config import ModelConfig
from omnifuser.models import (
    CrossModalFusionLayer,
    MultiResolutionTemporalExtractor,
    OmniFuser,
    sinusoidal_position_encoding,
)


def test_architecture_options_are_optional_and_reach_model_config() -> None:
    parser = build_parser()
    base = ["train", "--data-dir", "unused"]
    defaults = protocol_kwargs(parser.parse_args(base))["config_overrides"]
    assert not set(ARCHITECTURE_OPTIONS.values()) & set(defaults)

    args = parser.parse_args(
        base
        + [
            "--decomp-mode",
            "literal",
            "--rtd-layers",
            "3",
            "--rtd-stride",
            "2",
            "--rtd-rank",
            "4",
            "--gat-layers",
            "2",
            "--gat-heads",
            "2",
            "--fusion-layers",
            "3",
            "--landmarks",
            "6",
            "--fusion-rank",
            "16",
            "--dropout",
            "0.2",
        ]
    )
    overrides = protocol_kwargs(args)["config_overrides"]
    assert {field: overrides[field] for field in ARCHITECTURE_OPTIONS.values()} == {
        "decomp_mode": "literal",
        "num_rtd_layers": 3,
        "rtd_stride": 2,
        "rtd_rank": 4,
        "num_gat_layers": 2,
        "num_gat_heads": 2,
        "num_fusion_layers": 3,
        "num_landmarks": 6,
        "fusion_rank": 16,
        "dropout": 0.2,
    }
    config = ModelConfig(
        d_model=32,
        input_len=96,
        horizon=24,
        image_encoder="compact",
        prediction_head="direct",
        **overrides,
    )
    model = OmniFuser(config)
    assert isinstance(model.temporal_encoder, MultiResolutionTemporalExtractor)
    assert [block.output_steps for block in model.temporal_encoder.layers] == [
        48,
        24,
        12,
    ]
    assert len(model.image_branch.layers) == 2
    assert len(model.fusion_layers) == 3
    first_round = model.fusion_layers[0]
    assert isinstance(first_round, CrossModalFusionLayer)
    assert first_round.score_r.weight.shape == (6, 32)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_rtd_layers": 0}, "RTD"),
        ({"rtd_stride": 0}, "RTD"),
        ({"rtd_rank": 0}, "RTD"),
        ({"num_gat_layers": 0}, "GAT"),
        ({"num_gat_heads": 0}, "GAT"),
        ({"num_gat_heads": 3}, "divisible"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
        ({"dropout": float("nan")}, "dropout"),
    ],
)
def test_invalid_architecture_overrides_fail_early(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ModelConfig(**overrides)


def test_sinusoidal_encoding_has_alternating_sine_cosine_channels() -> None:
    encoding = sinusoidal_position_encoding(3, 4)
    assert encoding.shape == (3, 4)
    assert torch.equal(encoding[0], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert torch.isfinite(encoding).all()


@pytest.mark.parametrize("head", ["paper_literal", "lead_conditioned", "direct"])
def test_full_compact_model_produces_forecast_and_class_logits(head: str) -> None:
    config = ModelConfig(
        d_model=16,
        input_len=12,
        horizon=5,
        num_rtd_layers=2,
        rtd_stride=3,
        num_gat_layers=2,
        num_gat_heads=4,
        num_fusion_layers=2,
        prediction_head=head,
        image_encoder="compact",
        image_height=16,
        image_width=24,
    )
    model = OmniFuser(config).eval()
    force = torch.randn(2, 12, 3)
    images = torch.randn(2, 2, 3, 16, 24)
    indices = torch.tensor([[0] * 6 + [1] * 6, [1] * 12])
    with torch.no_grad():
        outputs = model(force, images, indices)
    assert outputs["force"].shape == (2, 5, 3)
    assert outputs["class_logits"].shape == (2, 5, 3)
    assert torch.isfinite(outputs["force"]).all()
    assert torch.isfinite(outputs["class_logits"]).all()


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], True),
        (["--dedup-identical-forces"], True),
        (["--dedup-identical-forces", "True"], True),
        (["--dedup-identical-forces", "False"], False),
        (["--dedup-identical-forces", "no"], False),
        (["--deduplicate-identical-forces", "false"], False),
        (["--keep-identical-forces"], False),
        (["--no-deduplicate-identical-forces"], False),
    ],
)
def test_dedup_argument_accepts_explicit_true_false_and_old_spellings(
    arguments: list[str], expected: bool
) -> None:
    import scripts.run_experiment_suite as suite

    parser = build_parser()
    assert parser.parse_args(["train", *arguments]).dedup_identical_forces is expected
    assert parser.parse_args(["inspect", *arguments]).dedup_identical_forces is expected
    assert suite.build_parser().parse_args(arguments).dedup_identical_forces is expected


def test_dedup_argument_rejects_ambiguous_values() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--dedup-identical-forces", "maybe"])


def test_image_gat_switch_sends_encoder_features_straight_to_fusion() -> None:
    """--no-image-gat (an experiment) removes the paper's image GAT; the default keeps it."""
    parser = build_parser()
    base = ["train", "--data-dir", "unused"]
    assert protocol_kwargs(parser.parse_args(base))["config_overrides"]["image_gat"] is True
    assert protocol_kwargs(parser.parse_args([*base, "--no-image-gat"]))["config_overrides"]["image_gat"] is False

    common = {"input_len": 24, "horizon": 12, "num_rtd_layers": 2, "image_encoder": "compact",
              "image_height": 32, "image_width": 96, "dropout": 0.0}
    assert len(OmniFuser(ModelConfig(**common)).image_branch.layers) == 4
    torch.manual_seed(0)
    model = OmniFuser(ModelConfig(**common, image_gat=False)).eval()
    branch = model.image_branch
    assert len(branch.layers) == 0
    images = torch.randn(2, 2, 3, 32, 96)
    indices = torch.cat([torch.zeros(2, 10, dtype=torch.long), torch.ones(2, 14, dtype=torch.long)], 1)
    with torch.no_grad():
        encoded = branch.encoder(images)
        gathered = torch.gather(encoded, 1, indices.unsqueeze(-1).expand(-1, -1, encoded.shape[-1]))
        expected = (gathered + branch.position_encoding[:24][None]
                    + branch.cycle.weight[0] + branch.batch.weight[0] + branch.relative.weight[0])
        assert torch.allclose(branch(images, indices), expected, atol=1e-6)
        outputs = model(torch.randn(2, 24, 3), images, indices)
    assert outputs["force"].shape == (2, 12, 3)
