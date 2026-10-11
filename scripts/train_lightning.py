from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnifuser.cross_validation import run_cross_validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full 10-fold leave-one-tool-out training and evaluation."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument(
        "--label-column", choices=("image_label", "tool_label"), default="image_label"
    )
    parser.add_argument(
        "--deduplicate-identical-forces",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-windows",
        type=int,
        help="optional cap for pipeline smoke tests; omit for full training",
    )
    parser.add_argument(
        "--max-eval-windows",
        type=int,
        help="optional cap for pipeline smoke tests; omit for full evaluation",
    )
    args = parser.parse_args()

    summary = run_cross_validation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        input_len=args.input_len,
        horizon=args.horizon,
        stride=args.stride,
        decimation=args.decimation,
        label_column=args.label_column,
        dedup_identical_forces=args.deduplicate_identical_forces,
        image_encoder=args.image_encoder,
        d_model=args.d_model,
        prediction_head=args.prediction_head,
        device_name=args.device,
        seed=args.seed,
        max_train_windows=args.max_train_windows,
        max_eval_windows=args.max_eval_windows,
    )
    print(json.dumps(summary["cv_mean_std"], indent=2))
    print(json.dumps(summary["out_of_fold"], indent=2))


if __name__ == "__main__":
    main()
