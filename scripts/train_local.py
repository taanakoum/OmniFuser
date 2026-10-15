from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnifuser.cli import (
    add_dedup_argument,
    add_eval_dedup_argument,
    add_protocol_arguments,
    protocol_kwargs,
)
from omnifuser.train import train_fold


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one local OmniFuser fold as an end-to-end training smoke test."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("Nonastreda_Data"))
    parser.add_argument("--test-tool", default="T10")
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_results/validation/local-smoke"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int, default=96, help="forecast steps (paper horizons go to 96)")
    parser.add_argument(
        "--stride",
        type=int,
        help="training-window stride; defaults to input length + horizon",
    )
    parser.add_argument(
        "--trim-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="trim pass lead-in/run-out with the Nonastreda span rule (default)",
    )
    parser.add_argument(
        "--image-encoder",
        choices=("efficientnet_b0", "compact"),
        default="efficientnet_b0",
    )
    parser.add_argument(
        "--prediction-head",
        choices=("paper_literal", "lead_conditioned", "direct"),
        default="direct",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the selected fold without smoke-test window limits",
    )
    add_dedup_argument(parser)
    add_eval_dedup_argument(parser)
    add_protocol_arguments(parser)
    args = parser.parse_args()

    result = train_fold(
        data_dir=args.data_dir,
        test_tool=args.test_tool,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device_name=args.device,
        horizon=args.horizon,
        stride=args.stride,
        trim_pass=args.trim_pass,
        dedup_identical_forces=args.dedup_identical_forces,
        eval_dedup_identical_forces=args.eval_dedup_identical_forces,
        image_encoder=args.image_encoder,
        prediction_head=args.prediction_head,
        max_train_windows=None if args.full else 2,
        max_eval_windows=None if args.full else 1,
        **protocol_kwargs(args),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
