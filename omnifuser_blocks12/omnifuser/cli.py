from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from omnifuser.data import (
    PASS_TRIM_REFERENCE_FRACTION,
    PASS_TRIM_THRESHOLD_FRACTION,
    PASS_TRIM_WINDOW_SAMPLES,
    load_nonastreda,
    make_tool_timelines,
)
from omnifuser.train import train_fold


def add_protocol_arguments(parser: argparse.ArgumentParser) -> None:
    """Options added by the alignment patch; shared by `omnifuser train` and the two scripts."""
    parser.add_argument(
        "--pool",
        type=int,
        default=1,
        help="average blocks of this many samples after trimming: one time step per POOL samples "
        "(the paper's step is much coarser than a 1 kHz sample; see the alignment review)",
    )
    parser.add_argument(
        "--image-lag",
        type=int,
        choices=(0, 1),
        default=0,
        help="0: each sample shows the photo of its own pass (paper-literal); "
        "1: the photo of the previous pass (causal)",
    )
    parser.add_argument(
        "--select-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="test the weights of the best validation epoch (default) instead of the last epoch",
    )
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="optimizer step every N batches (effective batch = batch size x N)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument(
        "--pre-attn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="self-attention before the shared/private split (not in the paper)",
    )
    parser.add_argument(
        "--als-mode",
        choices=("time", "k_sum", "k_colnorm"),
        default="time",
        help="landmark weights: softmax over time | printed Eq. 10 | printed Eq. 10 divided by column mass",
    )
    parser.add_argument(
        "--private-attn",
        choices=("residual_ln", "plain"),
        default="residual_ln",
        help="private path: LayerNorm(P + MHA(P)) (default, previous behaviour) or MHA(P) (printed text)",
    )
    parser.add_argument(
        "--share-c2f",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use the same C2F layer (weights) in every refinement round (Eq. 20 writes one G); default: one layer per round",
    )
    parser.add_argument("--modality-dropout", type=float, default=0.0,
                        help="probability that one modality is zeroed per training sample")
    parser.add_argument("--pretrained", action="store_true", help="ImageNet-pretrained EfficientNet-B0")
    parser.add_argument("--freeze-bn", action="store_true", help="keep BatchNorm layers in eval mode")
    # Architecture sizes of the paper's sensitivity study (Sec. VI-B3, Figs. 7-8).  Left unset they keep the
    # ModelConfig defaults (the paper's values), so existing runs and fold files are unaffected.
    parser.add_argument("--decomp-mode", choices=("branch", "literal"), default=None,
                        help="RTD: 'branch' = separate feed-forward per trend / seasonal branch (Fig. 5, default); "
                        "'literal' = U = T + S as printed in Eq. 3 (equals H_hat: the decomposition cancels)")
    for flag, help_text in (
        ("rtd-layers", "L, number of RTD blocks (paper 4)"),
        ("rtd-stride", "s, RTD downsampling stride and kernel (paper 3)"),
        ("rtd-rank", "r, rank of the trend projection P Q (paper 8)"),
        ("gat-layers", "number of graph-attention layers (paper 4)"),
        ("gat-heads", "heads per graph-attention layer and per attention in C2F (paper 4)"),
        ("fusion-layers", "n, number of C2F rounds (paper 2)"),
        ("landmarks", "k, adaptive landmarks (paper 4)"),
        ("fusion-rank", "rank of the low-rank fusion projection (paper 8)"),
    ):
        parser.add_argument(f"--{flag}", type=int, default=None, help=help_text)
    parser.add_argument("--dropout", type=float, default=None, help="dropout rate (paper 0.1)")


# command-line flag (argparse attribute) -> ModelConfig field
ARCHITECTURE_OPTIONS = {
    "decomp_mode": "decomp_mode",
    "rtd_layers": "num_rtd_layers",
    "rtd_stride": "rtd_stride",
    "rtd_rank": "rtd_rank",
    "gat_layers": "num_gat_layers",
    "gat_heads": "num_gat_heads",
    "fusion_layers": "num_fusion_layers",
    "landmarks": "num_landmarks",
    "fusion_rank": "fusion_rank",
    "dropout": "dropout",
}


def protocol_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Keyword arguments of train_fold / run_cross_validation for the options above."""
    overrides: dict[str, object] = {
        "pre_attn": args.pre_attn,
        "als_mode": args.als_mode,
        "private_attn": args.private_attn,
        "share_c2f": args.share_c2f,
        "modality_dropout": args.modality_dropout,
        "pretrained": args.pretrained,
        "freeze_bn": args.freeze_bn,
    }
    for attribute, field in ARCHITECTURE_OPTIONS.items():
        value = getattr(args, attribute, None)
        if value is not None:  # only what was asked for; the paper's values stay the defaults
            overrides[field] = value
    return {
        "pool": args.pool,
        "image_lag": args.image_lag,
        "select_best": args.select_best,
        "accumulation_steps": args.accumulation_steps,
        "num_workers": args.num_workers,
        "config_overrides": overrides,
    }


def _inspect(args: argparse.Namespace) -> None:
    records = load_nonastreda(
        args.data_dir,
        label_column=args.label_column,
        decimation=args.decimation,
        dedup_identical_forces=args.dedup_identical_forces,
        trim_pass=args.trim_pass,
        pool=args.pool,
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
    force_groups: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    duplicate_force_rows = 0
    for record in records:
        group = force_groups[(record.tool, record.run)]
        if any(np.array_equal(record.force, existing) for existing in group):
            duplicate_force_rows += 1
        else:
            group.append(record.force)
    timelines = make_tool_timelines(records)
    window_stride = (
        args.stride
        if args.stride is not None
        else args.input_len + args.horizon
    )
    if args.input_len < 1 or args.horizon < 1 or window_stride < 1:
        raise ValueError("input length, horizon, and stride must be positive")
    windows_by_tool = {
        timeline.tool: max(
            (len(timeline.force) - args.input_len - args.horizon) // window_stride + 1,
            0,
        )
        for timeline in timelines
    }
    most_trimmed_records = sorted(
        (
            record
            for record in records
            if record.trimmed_lead_samples + record.trimmed_runout_samples > 0
        ),
        key=lambda record: (
            record.trimmed_lead_samples + record.trimmed_runout_samples
        )
        / max(record.source_length_samples, 1),
        reverse=True,
    )[:10]
    report = {
        "records": len(records),
        "tools": len(by_tool),
        "classes": {name: counts[index] for index, name in enumerate(label_names)},
        "force_length_after_preprocessing": {
            "min": min(len(record.force) for record in records),
            "median": int(np.median([len(record.force) for record in records])),
            "max": max(len(record.force) for record in records),
        },
        "tool_stream_samples": {timeline.tool: len(timeline.force) for timeline in timelines},
        "retained_record_images": len({record.image_path for record in records}),
        "window": {
            "input_len": args.input_len,
            "horizon": args.horizon,
            "stride": window_stride,
            "windows_by_tool": windows_by_tool,
            "total_windows": sum(windows_by_tool.values()),
        },
        "exact_duplicate_B3_B4_pairs_remaining": duplicate_b3_b4,
        "exact_duplicate_force_rows_remaining": duplicate_force_rows,
        "decimation": args.decimation,
        "pool": args.pool,
        "dedup_identical_forces": args.dedup_identical_forces,
        "trim_pass": args.trim_pass,
        "pass_trimming": "span" if args.trim_pass else "none",
        "pass_trimming_parameters": (
            {
                "envelope_window_samples": PASS_TRIM_WINDOW_SAMPLES,
                "early_reference_fraction": PASS_TRIM_REFERENCE_FRACTION,
                "active_threshold_fraction": PASS_TRIM_THRESHOLD_FRACTION,
            }
            if args.trim_pass
            else None
        ),
        "pass_trimming_report": {
            "records_trimmed": sum(
                1
                for record in records
                if record.trimmed_lead_samples + record.trimmed_runout_samples > 0
            ),
            "removed_lead_samples": sum(
                record.trimmed_lead_samples for record in records
            ),
            "removed_runout_samples": sum(
                record.trimmed_runout_samples for record in records
            ),
            "most_trimmed_records": [
                {
                    "record_id": record.record_id,
                    "source_length_samples": record.source_length_samples,
                    "trimmed_lead_samples": record.trimmed_lead_samples,
                    "trimmed_runout_samples": record.trimmed_runout_samples,
                    "removed_fraction": (
                        record.trimmed_lead_samples + record.trimmed_runout_samples
                    )
                    / max(record.source_length_samples, 1),
                }
                for record in most_trimmed_records
            ],
        },
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
    inspect_parser.add_argument("--pool", type=int, default=1)
    inspect_parser.add_argument("--input-len", type=int, default=96)
    inspect_parser.add_argument("--horizon", type=int, default=24)
    inspect_parser.add_argument("--stride", type=int)
    inspect_parser.add_argument(
        "--trim-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="trim only pass lead-in/run-out with the Nonastreda span rule (default)",
    )
    duplicate_group = inspect_parser.add_mutually_exclusive_group()
    duplicate_group.add_argument(
        "--dedup-identical-forces",
        dest="dedup_identical_forces",
        action="store_true",
        help="merge exact force duplicates within each tool/run using the most-worn annotation (default)",
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
    train_parser.add_argument("--batch-size", type=int, default=32)
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
        "--trim-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="trim only pass lead-in/run-out with the Nonastreda span rule (default)",
    )
    train_parser.add_argument(
        "--label-column", choices=("image_label", "tool_label"), default="image_label"
    )
    duplicate_group = train_parser.add_mutually_exclusive_group()
    duplicate_group.add_argument(
        "--dedup-identical-forces",
        dest="dedup_identical_forces",
        action="store_true",
        help="merge exact force duplicates within each tool/run using the most-worn annotation (default)",
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
        choices=("paper_literal", "lead_conditioned", "direct"),
        default="lead_conditioned",
        help="paper-literal repeated head, lead-conditioned decoder, or direct (Linear d -> P x C, d -> P x 3)",
    )
    add_protocol_arguments(train_parser)
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
        trim_pass=args.trim_pass,
        image_encoder=args.image_encoder,
        d_model=args.d_model,
        prediction_head=args.prediction_head,
        max_train_windows=args.max_train_windows,
        max_eval_windows=args.max_eval_windows,
        device_name=args.device,
        seed=args.seed,
        **protocol_kwargs(args),
    )
    print(json.dumps(result, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
