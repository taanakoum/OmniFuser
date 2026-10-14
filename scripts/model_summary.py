"""Print OmniFuser stage shapes and parameter counts.

Examples:
    python scripts/model_summary.py --image-encoder compact --check
    python scripts/model_summary.py --horizon 96 --prediction-head direct --check
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnifuser.config import ModelConfig
from omnifuser.models import OmniFuser

EFFICIENTNET_B0_FEATURE_PARAMETERS = 4_007_548


def _parameter_count(*modules: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                total += parameter.numel()
    return total


def _expected_parameter_count(config: ModelConfig) -> int:
    """Count parameters independently from the instantiated modules."""
    d, horizon = config.d_model, config.horizon
    embedding = config.force_channels * d + d + d * d + d + 152 * d
    branch_ffn = 2 * d * d + 2 * d + 2 * d * d + d
    rtd = 0
    steps = config.input_len
    for _ in range(config.num_rtd_layers):
        down_steps = math.ceil(steps / config.rtd_stride)
        rtd += (
            d * config.rtd_stride
            + d * d
            + 2 * d * config.rtd_rank
            + 2 * branch_ffn
            + down_steps * config.input_len
            + config.input_len
            + 2 * (d * d + d)
        )
        steps = down_steps

    if config.image_encoder == "compact":
        image_encoder = (
            3 * 24 * 9
            + 2 * 24
            + 24 * 9 * 2
            + 2 * 48
            + 48 * 96
            + 2 * 96
            + 96 * d
            + d
        )
    else:
        image_encoder = EFFICIENTNET_B0_FEATURE_PARAMETERS + 1280 * d + d
    image = image_encoder + config.num_gat_layers * (d * d + 4 * d) + 152 * d

    attention = 4 * d * d + 4 * d
    fusion_layer = (
        2 * d * d
        + 2 * d * config.num_landmarks
        + 2 * d * d
        + 2 * attention
        + (4 * d if config.private_attn == "residual_ln" else 0)
        + 2 * (2 * d * d + d)
        + 2 * (6 * d + 2 * d)
        + 2 * (2 * d * d + d)
        + 2
        + 2 * d * config.fusion_rank
        + config.fusion_rank
        + config.fusion_rank * d
        + d
    )
    if config.pre_attn:
        fusion_layer += 2 * attention + 4 * d
    fusion = fusion_layer * (
        1 if config.share_c2f else config.num_fusion_layers
    )
    refinement = 2 * (config.num_fusion_layers - 1) * d * d
    if config.prediction_head == "direct":
        heads = (
            d * horizon * config.force_channels
            + horizon * config.force_channels
            + d * horizon * config.num_classes
            + horizon * config.num_classes
        )
    else:
        heads = (
            d * config.force_channels
            + config.force_channels
            + d * config.num_classes
            + config.num_classes
        )
        if config.prediction_head == "lead_conditioned":
            heads += horizon * d + 2 * d * d + d
    return embedding + rtd + image + fusion + refinement + heads


def _shape(output: Any) -> str:
    if isinstance(output, (tuple, list)):
        output = output[0]
    return "x".join(str(int(size)) for size in output.shape)


def summarize(config: ModelConfig, batch_size: int = 2) -> tuple[list[tuple[str, int, str]], int]:
    model = OmniFuser(config).eval()
    generator = np.random.default_rng(0)
    force = torch.as_tensor(
        generator.standard_normal((batch_size, config.input_len, config.force_channels)),
        dtype=torch.float32,
    )
    pictures = torch.as_tensor(
        generator.standard_normal(
            (batch_size, 2, 3, config.image_height, config.image_width)
        ),
        dtype=torch.float32,
    )
    indices = torch.zeros((batch_size, config.input_len), dtype=torch.long)
    indices[:, config.input_len // 2 :] = 1
    shapes: dict[str, str] = {}
    hooks = []

    def watch(name: str, module: torch.nn.Module) -> None:
        hooks.append(
            module.register_forward_hook(
                lambda _module, _args, output, key=name: shapes.__setitem__(
                    key, _shape(output)
                )
            )
        )

    watch("embedding", model.temporal_embedding)
    watch("mrte", model.temporal_encoder)
    watch("encoder", model.image_branch.encoder)
    watch("image", model.image_branch)
    for index, layer in enumerate(model.fusion_layers):
        watch(f"fusion-{index}", layer)
    try:
        with torch.inference_mode():
            output = model(force, pictures, indices)
    finally:
        for hook in hooks:
            hook.remove()

    rows = [
        (
            "Temporal embedding (Eq. 1)",
            _parameter_count(model.temporal_embedding),
            shapes["embedding"],
        ),
        (
            f"MRTE ({config.num_rtd_layers} RTD blocks; Eqs. 2-3)",
            _parameter_count(model.temporal_encoder),
            shapes["mrte"],
        ),
        (
            f"Image encoder ({config.image_encoder}; Eq. 4)",
            _parameter_count(model.image_branch.encoder),
            shapes["encoder"],
        ),
        (
            f"Image GAT and embeddings (Sec. V-B; {config.num_gat_layers} layers)",
            _parameter_count(model.image_branch)
            - _parameter_count(model.image_branch.encoder),
            shapes["image"],
        ),
    ]
    seen_layers: set[int] = set()
    for index, layer in enumerate(model.fusion_layers):
        shared = id(layer) in seen_layers
        seen_layers.add(id(layer))
        rows.append(
            (
                f"C2F round {index + 1}"
                + (" (shared weights)" if shared else ""),
                0 if shared else _parameter_count(layer),
                shapes[f"fusion-{index}"],
            )
        )
    rows.append(
        (
            f"Prediction heads ({config.prediction_head}; Eqs. 21-23)",
            _parameter_count(
                *(
                    module
                    for module in (
                        model.forecast_step_embedding,
                        model.forecast_projection,
                        getattr(model, "force_head", None),
                        getattr(model, "classification_head", None),
                        getattr(model, "direct_force", None),
                        getattr(model, "direct_class", None),
                    )
                    if module is not None
                )
            ),
            f"force {tuple(output['force'].shape)}, logits {tuple(output['class_logits'].shape)}",
        )
    )
    return rows, _parameter_count(model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-encoder", choices=("efficientnet_b0", "compact"), default="efficientnet_b0")
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--prediction-head", choices=("paper_literal", "lead_conditioned", "direct"), default="direct")
    parser.add_argument("--share-c2f", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--check", action="store_true", help="compare the instantiated count with the dimension-derived count")
    args = parser.parse_args(argv)

    config = ModelConfig(
        d_model=args.d_model,
        input_len=args.input_len,
        horizon=args.horizon,
        num_fusion_layers=args.rounds,
        prediction_head=args.prediction_head,
        image_encoder=args.image_encoder,
        share_c2f=args.share_c2f,
    )
    rows, actual = summarize(config, args.batch_size)
    print(f"{'stage':58s} {'parameters':>12s}  output shape")
    for stage, count, shape in rows:
        print(f"{stage:58s} {count:12,d}  {shape}")
    print(f"{'total':58s} {actual:12,d}")
    if args.check:
        expected = _expected_parameter_count(config)
        matches = expected == actual
        print(
            f"implied by the dimensions: {expected:,d} "
            f"({'match' if matches else 'DIFFERENT'})"
        )
        if not matches:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
