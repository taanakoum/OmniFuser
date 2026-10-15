from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from omnifuser.data import (
    PASS_TRIM_REFERENCE_FRACTION,
    PASS_TRIM_THRESHOLD_FRACTION,
    PASS_TRIM_WINDOW_SAMPLES,
    load_nonastreda,
    make_tool_timelines,
)
from omnifuser.train import train_fold


def parse_bool(value: str) -> bool:
    """Parse an explicit command-line boolean such as True/False, yes/no, or 1/0."""
    normalized = value.strip().lower()
    if normalized in {"true", "t", "yes", "y", "1", "on"}:
        return True
    if normalized in {"false", "f", "no", "n", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected True or False, got {value!r}")


def add_dedup_argument(parser: argparse.ArgumentParser) -> None:
    """``--dedup-identical-forces {True,False}``; a bare flag means True.

    The older spellings stay as aliases: ``--deduplicate-identical-forces``
    (with an optional value), and ``--keep-identical-forces`` /
    ``--no-deduplicate-identical-forces`` (both mean False).
    """
    parser.add_argument(
        "--dedup-identical-forces",
        "--deduplicate-identical-forces",
        dest="dedup_identical_forces",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        metavar="{True,False}",
        help=(
            "merge bit-identical force recordings within each tool/run, keeping the "
            "most-worn annotation (True, default) or keep every record (False)"
        ),
    )
    parser.add_argument(
        "--keep-identical-forces",
        "--no-deduplicate-identical-forces",
        dest="dedup_identical_forces",
        action="store_false",
        help="same as --dedup-identical-forces False",
    )


def add_eval_dedup_argument(parser: argparse.ArgumentParser) -> None:
    """``--eval-dedup-identical-forces {True,False}`` for the validation/test data.

    Omitted, it follows ``--dedup-identical-forces``. Setting it to True scores a
    ``--dedup-identical-forces False`` run on the same deduplicated windows as a
    default run, so the two can be compared on identical test data.
    """
    parser.add_argument(
        "--eval-dedup-identical-forces",
        dest="eval_dedup_identical_forces",
        type=parse_bool,
        default=None,
        metavar="{True,False}",
        help=(
            "duplicate handling for validation and test windows "
            "(default: same as --dedup-identical-forces)"
        ),
    )


def add_protocol_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ablation",
        choices=(
            "none",
            "without_rr",
            "without_mrte",
            "without_rr_mrte",
            "concat_c2f",
            "without_c2f",
            "without_als",
            "without_hybrid_gate",
            "moving_average",
        ),
        default="none",
        help="paper experiment variant; 'none' is the full configured model",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=1,
        help="mean-pool this many samples per time step after trimming and decimation",
    )
    parser.add_argument(
        "--image-lag",
        type=int,
        choices=(0, 1),
        default=0,
        help="use the current pass photo (0) or preceding record's photo in tool order (1)",
    )
    parser.add_argument(
        "--select-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="test the best validation-loss epoch instead of the last epoch",
    )
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume interrupted folds and reuse complete matching CV folds",
    )
    parser.add_argument(
        "--pre-attn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include extra self-attention before shared/private separation (not in paper figures)",
    )
    parser.add_argument(
        "--als-mode",
        choices=("time", "k_sum", "k_colnorm"),
        default="k_colnorm",
        help="landmark weighting: Eq. 10 with column normalization (default), printed Eq. 10 sums, or time-softmax",
    )
    parser.add_argument(
        "--private-attn",
        choices=("residual_ln", "plain"),
        default="plain",
        help="private path: MHA(P) as printed (default) or LayerNorm(P + MHA(P))",
    )
    parser.add_argument(
        "--share-c2f",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="reuse C2F weights across refinement rounds; paper Eq. 20 uses one G, figure shows layered blocks",
    )
    parser.add_argument(
        "--exact-split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use the Eq. 7 projection c*S as the shared part so Z = P + S exactly",
    )
    parser.add_argument("--modality-dropout", type=float, default=0.0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-bn", action="store_true")
    parser.add_argument(
        "--decomp-mode",
        choices=("branch", "literal"),
        default=None,
        help="RTD decomposition: separate branch feed-forwards or literal Eq. 3 sum",
    )
    for flag, help_text in (
        ("rtd-layers", "number of RTD blocks (paper default: 4)"),
        ("rtd-stride", "RTD temporal kernel and stride (paper default: 3)"),
        ("rtd-rank", "rank of the RTD trend projection (paper default: 8)"),
        ("gat-layers", "number of image graph-attention layers (paper default: 4)"),
        ("gat-heads", "attention heads in GAT and C2F (paper default: 4)"),
        ("fusion-layers", "number of C2F refinement rounds (paper default: 2)"),
        ("landmarks", "number of adaptive landmarks (paper default: 4)"),
        ("fusion-rank", "rank of the low-rank fusion projection (paper default: 8)"),
    ):
        parser.add_argument(f"--{flag}", type=int, default=None, help=help_text)
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="model dropout rate (paper default: 0.1)",
    )


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


class ProtocolKwargs(TypedDict):
    """Keyword arguments shared by ``train_fold`` and ``run_cross_validation``."""

    pool: int
    image_lag: int
    select_best: bool
    accumulation_steps: int
    num_workers: int
    resume: bool
    config_overrides: dict[str, Any]


def protocol_kwargs(args: argparse.Namespace) -> ProtocolKwargs:
    overrides: dict[str, Any] = {
        "ablation": args.ablation,
        "pre_attn": args.pre_attn,
        "als_mode": args.als_mode,
        "private_attn": args.private_attn,
        "share_c2f": args.share_c2f,
        "exact_split": args.exact_split,
        "modality_dropout": args.modality_dropout,
        "pretrained": args.pretrained,
        "freeze_bn": args.freeze_bn,
    }
    for attribute, field in ARCHITECTURE_OPTIONS.items():
        value = getattr(args, attribute, None)
        if value is not None:
            overrides[field] = value
    return {
        "pool": args.pool,
        "image_lag": args.image_lag,
        "select_best": args.select_best,
        "accumulation_steps": args.accumulation_steps,
        "num_workers": args.num_workers,
        "resume": args.resume,
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
        "pool_dropped_samples": {
            "per_record_sum": sum(record.pool_dropped_samples for record in records),
            "unit": "post-decimation source samples",
        },
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
    inspect_parser.add_argument("--horizon", type=int, default=96)
    inspect_parser.add_argument("--stride", type=int)
    inspect_parser.add_argument(
        "--trim-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="trim only pass lead-in/run-out with the Nonastreda span rule (default)",
    )
    add_dedup_argument(inspect_parser)
    inspect_parser.set_defaults(handler=_inspect)

    train_parser = subparsers.add_parser("train", help="train and evaluate one tool-held-out fold")
    train_parser.add_argument("--data-dir", type=Path, default=Path("Nonastreda_Data"))
    train_parser.add_argument("--test-tool", default="T10")
    train_parser.add_argument("--output-dir", type=Path, default=Path("experiment_results/train"))
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--input-len", type=int, default=96)
    train_parser.add_argument("--horizon", type=int, default=96)
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
    add_dedup_argument(train_parser)
    add_eval_dedup_argument(train_parser)
    train_parser.add_argument(
        "--image-encoder",
        choices=("efficientnet_b0", "compact"),
        default="efficientnet_b0",
    )
    train_parser.add_argument("--d-model", type=int, default=32)
    train_parser.add_argument(
        "--prediction-head",
        choices=("paper_literal", "lead_conditioned", "direct"),
        default="direct",
        help="direct per-lead Eq. 21-22 head (default), paper-literal repeated, or lead-conditioned",
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
        eval_dedup_identical_forces=args.eval_dedup_identical_forces,
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
