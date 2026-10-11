"""Parameter table and tensor shapes of OmniFuser, stage by stage, with the paper equation each stage implements.

    python scripts/model_summary.py                                  # paper settings, EfficientNet-B0, horizon 24
    python scripts/model_summary.py --horizon 96 --prediction-head direct
    python scripts/model_summary.py --image-encoder compact          # no torchvision needed
    python scripts/model_summary.py --share-c2f --rounds 3

It builds the model from ``ModelConfig``, runs one random batch through it (eval mode, CPU) and prints, for each stage,
the number of trainable parameters (a shared C2F layer is counted once) and the shape of its output.  With
``--check`` it also compares the total with the number implied by the dimensions alone (tests/frontend_reference.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnifuser.config import ModelConfig  # noqa: E402
from omnifuser.models import OmniFuser  # noqa: E402


def _shape(output: Any) -> str:
    if isinstance(output, (tuple, list)):
        output = output[0]
    return "x".join(str(int(n)) for n in output.shape)


def parameter_count(*modules: Any) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                total += int(np.prod(tuple(parameter.shape)))
    return total


def summarize(config: ModelConfig, batch: int = 2, images: int = 2, height: int | None = None, width: int | None = None):
    """Returns (rows, total): rows = (stage, equation, parameters, output shape)."""
    model = OmniFuser(config).eval()
    height, width = height or config.image_height, width or config.image_width
    steps, generator = config.input_len, np.random.default_rng(0)
    force = torch.as_tensor(generator.standard_normal((batch, steps, config.force_channels)), dtype=torch.float32)
    pictures = torch.as_tensor(generator.standard_normal((batch, images, 3, height, width)), dtype=torch.float32)
    indices = torch.as_tensor(np.tile(np.minimum(np.arange(steps) * images // steps, images - 1), (batch, 1)), dtype=torch.long)

    shapes: dict[str, str] = {}
    hooks = []

    def watch(name: str, module: Any) -> None:
        hooks.append(module.register_forward_hook(lambda mod, args, out, key=name: shapes.__setitem__(key, _shape(out))))

    watch("embedding", model.temporal_embedding)
    watch("mrte", model.temporal_encoder)
    watch("encoder", model.image_branch.encoder)
    watch("image", model.image_branch)
    for index, layer in enumerate(model.fusion_layers):
        watch(f"c2f{index}", layer)
    for index, module in enumerate(model.refine_sensor):
        watch(f"refine{index}", module)
    with torch.no_grad():
        outputs = model(force, pictures, indices)
    for handle in hooks:
        handle.remove()

    rows: list[tuple[str, str, int, str]] = [
        ("temporal embedding", "Eq. 1", parameter_count(model.temporal_embedding), shapes["embedding"]),
        (f"MRTE ({config.num_rtd_layers} RTD blocks)", "Eq. 2-3", parameter_count(model.temporal_encoder), shapes["mrte"]),
        (f"image encoder ({config.image_encoder})", "Eq. 4", parameter_count(model.image_branch.encoder), shapes["encoder"]),
        (f"image branch GAT {config.num_gat_layers} x {config.num_gat_heads} + position + period", "Sec. V-B",
         parameter_count(model.image_branch) - parameter_count(model.image_branch.encoder), shapes["image"]),
    ]
    seen_layers: set[int] = set()
    for index, layer in enumerate(model.fusion_layers):
        shared = id(layer) in seen_layers
        seen_layers.add(id(layer))
        rows.append((f"C2F round {index + 1}" + (" (same layer as an earlier round)" if shared else ""), "Eq. 5, 10-11, 16-19",
                     0 if shared else parameter_count(layer), shapes[f"c2f{index}"]))
        if index:
            rows.append((f"  refinement maps P^r, P^i before round {index + 1}", "Eq. 20",
                         parameter_count(model.refine_sensor[index - 1], model.refine_image[index - 1]),
                         shapes[f"refine{index - 1}"]))
    head_modules = [m for m in (model.forecast_step_embedding, model.forecast_projection, getattr(model, "force_head", None),
                                getattr(model, "classification_head", None), getattr(model, "direct_force", None),
                                getattr(model, "direct_class", None)) if m is not None]
    rows.append((f"heads ({config.prediction_head})", "Eq. 21-23", parameter_count(*head_modules),
                 f"force {_shape(outputs['force'])}, logits {_shape(outputs['class_logits'])}"))
    total = parameter_count(model)
    return rows, total


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-encoder", choices=("efficientnet_b0", "compact"), default="efficientnet_b0")
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=2, help="n in Eq. 20 (number of C2F rounds)")
    parser.add_argument("--prediction-head", choices=("paper_literal", "lead_conditioned", "direct"), default="lead_conditioned")
    parser.add_argument("--share-c2f", action="store_true")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--images", type=int, default=2, help="distinct photos per window")
    parser.add_argument("--height", type=int, help="photo height fed to the CNN (default: config, 128)")
    parser.add_argument("--width", type=int, help="photo width (default: config, 384)")
    parser.add_argument("--check", action="store_true", help="compare the total with the count implied by the dimensions")
    args = parser.parse_args(argv)

    config = ModelConfig(d_model=args.d_model, input_len=args.input_len, horizon=args.horizon, num_fusion_layers=args.rounds,
                         prediction_head=args.prediction_head, image_encoder=args.image_encoder, share_c2f=args.share_c2f)
    rows, total = summarize(config, args.batch, args.images, args.height, args.width)
    print(f"{'stage':62s} {'equation':20s} {'parameters':>11s}  output")
    for stage, equation, count, shape in rows:
        print(f"{stage:62s} {equation:20s} {count:11,d}  {shape}")
    print(f"{'total':62s} {'':20s} {total:11,d}")
    result: dict[str, Any] = {"rows": rows, "total": total}
    if args.check:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        import frontend_reference as ref

        expected = ref.expected_parameter_count(
            d=config.d_model, input_len=config.input_len, horizon=config.horizon, channels=config.force_channels,
            classes=config.num_classes, rtd_layers=config.num_rtd_layers, stride=config.rtd_stride, rank=config.rtd_rank,
            gat_layers=config.num_gat_layers, heads=config.num_gat_heads, rounds=config.num_fusion_layers,
            landmarks=config.num_landmarks, fusion_rank=config.fusion_rank, share_c2f=config.share_c2f,
            pre_attn=config.pre_attn, private_attn=config.private_attn, head=config.prediction_head,
            encoder="compact" if config.image_encoder == "compact" else "efficientnet_b0")
        print(f"implied by the dimensions: {expected['total']:,d} ({'match' if expected['total'] == total else 'DIFFERENT'})")
        result["expected"] = expected
    return result


if __name__ == "__main__":
    main()
