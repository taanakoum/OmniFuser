from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from omnifuser.data import load_nonastreda, make_tool_timelines
from omnifuser.train import train_fold


def _inspect(args: argparse.Namespace) -> None:
    records = load_nonastreda(
        args.data_dir,
        label_column=args.label_column,
        decimation=args.decimation,
        dedup_identical_forces=args.dedup_identical_forces,
    )
    label_names = (
        ("sharp", "used", "dulled")
        if args.label_column == "image_label"
        else ("1", "2", "3")
    )
    counts = Counter(record.label for record in records)
    by_tool: dict[str, list[object]] = defaultdict(list)
    for record in records:
        by_tool[record.tool].append(record)
    pairs: dict[tuple[str, int], dict[int, np.ndarray]] = defaultdict(dict)
    for record in records:
        pairs[(record.tool, record.run)][record.block] = record.force
    duplicate_b3_b4 = sum(
        1
        for blocks in pairs.values()
        if 3 in blocks and 4 in blocks and np.array_equal(blocks[3], blocks[4])
    )
    timelines = make_tool_timelines(records)
    report = {
        "records": len(records),
        "tools": len(by_tool),
        "classes": {name: counts[index] for index, name in enumerate(label_names)},
        "force_length_after_decimation": {
            "min": min(len(record.force) for record in records),
            "median": int(np.median([len(record.force) for record in records])),
            "max": max(len(record.force) for record in records),
        },
        "tool_stream_samples": {timeline.tool: len(timeline.force) for timeline in timelines},
        "exact_duplicate_B3_B4_pairs_remaining": duplicate_b3_b4,
        "decimation": args.decimation,
        "dedup_identical_forces": args.dedup_identical_forces,
        "duplicate_policy": (
            "retain record with greatest flank_wear"
            if args.dedup_identical_forces
            else "keep all records"
        ),
        "label_column": args.label_column,
    }
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnifuser")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="validate and summarize Nonastreda")
    inspect_parser.add_argument("--data-dir", type=Path, default=Path("Nonastreda_Data"))
    inspect_parser.add_argument(
        "--label-column", choices=("image_label", "tool_label"), default="image_label"
    )
    inspect_parser.add_argument("--decimation", type=int, default=1)
    duplicate_group = inspect_parser.add_mutually_exclusive_group()
    duplicate_group.add_argument(
        "--dedup-identical-forces",
        dest="dedup_identical_forces",
        action="store_true",
        help="merge exact B3/B4 force duplicates using the more-worn record annotation (default)",
    )
    duplicate_group.add_argument(
        "--keep-identical-forces",
        dest="dedup_identical_forces",
        action="store_false",
        help="retain both records even when their force arrays are identical",
    )
    inspect_parser.set_defaults(dedup_identical_forces=True)
    inspect_parser.set_defaults(handler=_inspect)

    train_parser = subparsers.add_parser("train", help="train and evaluate one tool-held-out fold")
    train_parser.add_argument("--data-dir", type=Path, default=Path("Nonastreda_Data"))
    train_parser.add_argument("--test-tool", default="T10")
    train_parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--input-len", type=int, default=96)
    train_parser.add_argument("--horizon", type=int, default=24)
    train_parser.add_argument(
        "--stride",
        type=int,
        help="training-window stride; defaults to input length + forecast horizon",
    )
    train_parser.add_argument("--decimation", type=int, default=1)
    train_parser.add_argument(
        "--label-column", choices=("image_label", "tool_label"), default="image_label"
    )
    duplicate_group = train_parser.add_mutually_exclusive_group()
    duplicate_group.add_argument(
        "--dedup-identical-forces",
        dest="dedup_identical_forces",
        action="store_true",
        help="merge exact B3/B4 force duplicates using the more-worn record annotation (default)",
    )
    duplicate_group.add_argument(
        "--keep-identical-forces",
        dest="dedup_identical_forces",
        action="store_false",
        help="retain both records even when their force arrays are identical",
    )
    train_parser.set_defaults(dedup_identical_forces=True)
    train_parser.add_argument(
        "--image-encoder",
        choices=("efficientnet_b0", "compact"),
        default="efficientnet_b0",
    )
    train_parser.add_argument("--d-model", type=int, default=32)
    train_parser.add_argument(
        "--prediction-head",
        choices=("paper_literal", "lead_conditioned"),
        default="lead_conditioned",
        help="paper-literal repeated head or lead-conditioned multi-horizon head",
    )
    train_parser.add_argument("--max-train-windows", type=int)
    train_parser.add_argument("--max-eval-windows", type=int)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(handler=_train)
    return parser


def _train(args: argparse.Namespace) -> None:
    result = train_fold(
        data_dir=args.data_dir,
        test_tool=args.test_tool,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        input_len=args.input_len,
        horizon=args.horizon,
        stride=args.stride,
        decimation=args.decimation,
        label_column=args.label_column,
        dedup_identical_forces=args.dedup_identical_forces,
        image_encoder=args.image_encoder,
        d_model=args.d_model,
        prediction_head=args.prediction_head,
        max_train_windows=args.max_train_windows,
        max_eval_windows=args.max_eval_windows,
        device_name=args.device,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
