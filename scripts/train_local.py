from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnifuser.train import train_fold


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one local OmniFuser fold as an end-to-end training smoke test."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("Nonastreda_Data"))
    parser.add_argument("--test-tool", default="T10")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/local-smoke"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--image-encoder",
        choices=("efficientnet_b0", "compact"),
        default="efficientnet_b0",
    )
    parser.add_argument(
        "--prediction-head",
        choices=("paper_literal", "lead_conditioned"),
        default="lead_conditioned",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the selected fold without smoke-test window limits",
    )
    args = parser.parse_args()

    result = train_fold(
        data_dir=args.data_dir,
        test_tool=args.test_tool,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device_name=args.device,
        image_encoder=args.image_encoder,
        prediction_head=args.prediction_head,
        max_train_windows=None if args.full else 2,
        max_eval_windows=None if args.full else 1,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
