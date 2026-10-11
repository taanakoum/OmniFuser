from __future__ import annotations

import csv
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from scipy.signal import decimate
from torch import Tensor
from torch.utils.data import Dataset


CLASS_NAMES = ("sharp", "used", "dulled")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
RECORD_ID = re.compile(r"^T(?P<tool>\d+)R(?P<run>\d+)B(?P<block>\d+)$")
IMAGE_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
PASS_TRIM_WINDOW_SAMPLES = 20
PASS_TRIM_REFERENCE_FRACTION = 0.05
PASS_TRIM_THRESHOLD_FRACTION = 0.5


def nonastreda_source_signature(data_dir: str | Path) -> dict[str, object]:
    """Return a lightweight file manifest for safe fold/resume reuse."""
    root = Path(data_dir).resolve()
    files = [
        root / "forces_xyz_raw.mat",
        root / "labels.csv",
        root / "labels_reg.csv",
        *sorted((root / "tool").glob("*.jpg")),
    ]
    manifest: list[dict[str, object]] = []
    for path in files:
        stat = path.stat()
        manifest.append(
            {
                "path": str(path.relative_to(root)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {"root": str(root), "files": manifest}


@dataclass(frozen=True)
class Record:
    record_id: str
    tool: str
    run: int
    block: int
    force: np.ndarray
    label: int
    image_path: Path
    source_length_samples: int = 0
    trimmed_lead_samples: int = 0
    trimmed_runout_samples: int = 0
    pool_dropped_samples: int = 0


@dataclass(frozen=True)
class ToolTimeline:
    tool: str
    force: np.ndarray
    labels: np.ndarray
    image_codes: np.ndarray
    cycle_codes: np.ndarray
    batch_codes: np.ndarray
    relative_codes: np.ndarray
    image_paths: tuple[Path, ...]
    image_labels: np.ndarray | None = None


def _block_mean(force: np.ndarray, factor: int) -> np.ndarray:
    """Average complete consecutive blocks; discard an incomplete trailing block."""
    if factor < 1:
        raise ValueError("pool factor must be at least 1")
    if factor == 1:
        return force
    steps = len(force) // factor
    if steps < 1:
        raise ValueError(f"record of {len(force)} samples is shorter than pool={factor}")
    blocks = force[: steps * factor].reshape(steps, factor, force.shape[1])
    return blocks.mean(axis=1, dtype=np.float64).astype(np.float32)


def pool_records(records: Sequence[Record], factor: int) -> list[Record]:
    """Coarsen each record independently, preserving record/image/label ownership."""
    pooled: list[Record] = []
    for record in records:
        dropped = len(record.force) % factor if factor > 0 else 0
        pooled.append(
            replace(
                record,
                force=_block_mean(record.force, factor),
                pool_dropped_samples=dropped,
            )
        )
    return pooled


@dataclass(frozen=True)
class ToolFold:
    train_tools: tuple[str, ...]
    validation_tools: tuple[str, ...]
    test_tools: tuple[str, ...]


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows or "id" not in rows[0]:
        raise ValueError(f"{path} must have a header and an id column")
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path} contains duplicate record ids")
    return result


def _decode_record_id(filename: str) -> tuple[str, int, int, str]:
    record_id = Path(filename).stem
    match = RECORD_ID.fullmatch(record_id)
    if match is None:
        raise ValueError(f"Unexpected Nonastreda record id: {record_id!r}")
    return (
        f"T{int(match.group('tool'))}",
        int(match.group("run")),
        int(match.group("block")),
        record_id,
    )


def _nonastreda_span_bounds(force: np.ndarray) -> tuple[int, int]:
    window_samples = PASS_TRIM_WINDOW_SAMPLES
    if len(force) < window_samples:
        return 0, len(force)
    magnitude = np.square(force.astype(np.float64, copy=False)).sum(axis=1)
    envelope = np.sqrt(
        uniform_filter1d(magnitude, size=window_samples, mode="nearest")
    )
    reference_length = max(
        window_samples, int(len(force) * PASS_TRIM_REFERENCE_FRACTION)
    )
    reference_level = float(np.median(envelope[:reference_length]))
    if not np.isfinite(reference_level) or reference_level <= np.finfo(float).eps:
        return 0, len(force)
    active = np.flatnonzero(
        envelope >= reference_level * PASS_TRIM_THRESHOLD_FRACTION
    )
    if active.size == 0:
        return 0, len(force)
    return int(active[0]), int(active[-1]) + 1


def load_nonastreda(
    data_dir: str | Path,
    *,
    label_column: str = "image_label",
    decimation: int = 1,
    dedup_identical_forces: bool = True,
    trim_pass: bool = True,
    pool: int = 1,
) -> list[Record]:
    """Load aligned records, optionally trim edges and mean-pool each pass."""
    root = Path(data_dir)
    if label_column not in {"image_label", "tool_label"}:
        raise ValueError("label_column must be 'image_label' or 'tool_label'")
    if decimation < 1 or pool < 1:
        raise ValueError("decimation and pool must be at least 1")
    labels = _read_csv(root / "labels.csv")
    regression = _read_csv(root / "labels_reg.csv")
    if set(regression) != set(labels):
        missing = set(labels) - set(regression)
        extra = set(regression) - set(labels)
        raise ValueError(
            "labels_reg.csv ids must match labels.csv "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    flank_wear: dict[str, float] = {}
    for record_id, row in regression.items():
        try:
            value = float(row["flank_wear"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"invalid flank_wear value for {record_id}") from error
        if not np.isfinite(value):
            raise ValueError(f"non-finite flank_wear value for {record_id}")
        flank_wear[record_id] = value

    mat_path = root / "forces_xyz_raw.mat"
    contents = loadmat(mat_path, variable_names=["baseDatastore"], simplify_cells=True)
    if "baseDatastore" not in contents:
        raise ValueError(f"{mat_path} does not contain the expected 'baseDatastore' variable")
    datastore = contents["baseDatastore"]
    if not isinstance(datastore, np.ndarray) or datastore.ndim != 2 or datastore.shape[1] < 4:
        raise ValueError("baseDatastore must be a table with filename, labels, and force data")

    records: list[Record] = []
    seen_ids: set[str] = set()
    missing_images: list[str] = []
    raw_forces: dict[str, np.ndarray] = {}
    for row in datastore:
        _, _, _, record_id = _decode_record_id(str(row[0]))
        raw_forces[record_id] = np.asarray(row[3])
    if set(raw_forces) != set(labels):
        missing = set(labels) - set(raw_forces)
        extra = set(raw_forces) - set(labels)
        raise ValueError(
            "force record ids must match labels.csv "
            f"(missing={len(missing)}, extra={len(extra)})"
        )

    discarded_duplicate_ids: set[str] = set()
    if dedup_identical_forces:
        records_by_run: dict[tuple[str, int], list[str]] = {}
        for record_id in raw_forces:
            tool, run, _, _ = _decode_record_id(record_id)
            records_by_run.setdefault((tool, run), []).append(record_id)
        for record_ids in records_by_run.values():
            force_groups: list[list[str]] = []
            for record_id in sorted(
                record_ids, key=lambda item: _decode_record_id(item)[2]
            ):
                raw_force = raw_forces[record_id]
                matching_group = next(
                    (
                        group
                        for group in force_groups
                        if raw_forces[group[0]].shape == raw_force.shape
                        and np.array_equal(raw_forces[group[0]], raw_force)
                    ),
                    None,
                )
                if matching_group is None:
                    force_groups.append([record_id])
                else:
                    matching_group.append(record_id)
            for group in force_groups:
                if len(group) < 2:
                    continue
                retained_id = max(
                    group,
                    key=lambda item: (
                        flank_wear[item],
                        -_decode_record_id(item)[2],
                    ),
                )
                discarded_duplicate_ids.update(
                    item for item in group if item != retained_id
                )

    for row in datastore:
        filename = str(row[0])
        tool, run, block, record_id = _decode_record_id(filename)
        if record_id in seen_ids:
            raise ValueError(f"duplicate force record {record_id}")
        seen_ids.add(record_id)
        if record_id not in labels:
            raise ValueError(f"{record_id} is missing from labels.csv")
        csv_label = labels[record_id].get(label_column, "").strip().lower()
        mat_label = str(row[1]).strip().lower()
        if label_column == "image_label" and csv_label != mat_label:
            raise ValueError(f"label mismatch for {record_id}: MAT={mat_label}, CSV={csv_label}")
        if label_column == "image_label":
            if csv_label not in CLASS_TO_INDEX:
                raise ValueError(f"unknown image label {csv_label!r} for {record_id}")
            class_index = CLASS_TO_INDEX[csv_label]
        else:
            try:
                class_index = int(csv_label) - 1
            except ValueError as error:
                raise ValueError(f"invalid tool label {csv_label!r} for {record_id}") from error
            if class_index not in range(3):
                raise ValueError(f"tool label for {record_id} must be 1, 2, or 3")

        image_path = root / "tool" / filename
        if not image_path.is_file():
            missing_images.append(str(image_path))
        raw_force = np.asarray(row[3])
        if raw_force.ndim != 2 or raw_force.shape[0] != 3:
            raise ValueError(
                f"{record_id} force data must have shape (3, samples), got {raw_force.shape}"
            )
        if not np.isfinite(raw_force).all():
            raise ValueError(f"{record_id} force data contains non-finite values")
        if record_id in discarded_duplicate_ids:
            continue
        force = np.ascontiguousarray(raw_force.T.astype(np.float32, copy=False))
        source_length = len(force)
        trimmed_lead = 0
        trimmed_runout = 0
        if trim_pass:
            start, end = _nonastreda_span_bounds(force)
            trimmed_lead = start
            trimmed_runout = source_length - end
            force = force[start:end]
        if decimation > 1:
            force = decimate(force, decimation, axis=0, zero_phase=True).astype(
                np.float32, copy=False
            )
        pool_dropped_samples = len(force) % pool
        force = _block_mean(force, pool)
        records.append(
            Record(
                record_id=record_id,
                tool=tool,
                run=run,
                block=block,
                force=force,
                label=class_index,
                image_path=image_path,
                source_length_samples=source_length,
                trimmed_lead_samples=trimmed_lead,
                trimmed_runout_samples=trimmed_runout,
                pool_dropped_samples=pool_dropped_samples,
            )
        )
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} image(s), including {missing_images[0]}")
    extra_labels = set(labels) - seen_ids
    if extra_labels:
        raise ValueError(f"labels.csv has {len(extra_labels)} id(s) without force records")
    if len(records) == 0:
        raise ValueError("No valid records found in the dataset")
    return sorted(records, key=lambda item: (int(item.tool[1:]), item.run, item.block))


def make_tool_timelines(
    records: Sequence[Record], image_lag: int = 0
) -> list[ToolTimeline]:
    """Build tool streams; optionally use the preceding record's image while labels stay current."""
    if image_lag not in (0, 1):
        raise ValueError("image_lag must be 0 or 1")
    timelines: list[ToolTimeline] = []
    for tool in sorted({record.tool for record in records}, key=lambda item: int(item[1:])):
        selected = sorted(
            (record for record in records if record.tool == tool),
            key=lambda item: (item.run, item.block),
        )
        image_paths = tuple(dict.fromkeys(record.image_path for record in selected))
        image_code = {path: index for index, path in enumerate(image_paths)}
        force_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        image_parts: list[np.ndarray] = []
        cycle_parts: list[np.ndarray] = []
        batch_parts: list[np.ndarray] = []
        relative_parts: list[np.ndarray] = []
        image_label_parts: list[np.ndarray] = []
        for position, record in enumerate(selected):
            length = len(record.force)
            shown_record = selected[max(position - image_lag, 0)]
            force_parts.append(record.force)
            label_parts.append(np.full(length, record.label, dtype=np.int8))
            image_parts.append(
                np.full(length, image_code[shown_record.image_path], dtype=np.int16)
            )
            image_label_parts.append(
                np.full(length, shown_record.label, dtype=np.int8)
            )
            cycle_parts.append(np.full(length, record.run, dtype=np.int16))
            batch_parts.append(np.full(length, record.block, dtype=np.int16))
            relative_parts.append(
                np.minimum(np.arange(length, dtype=np.int32) * 16 // length, 15).astype(
                    np.int16
                )
            )
        timelines.append(
            ToolTimeline(
                tool=tool,
                force=np.concatenate(force_parts).astype(np.float32, copy=False),
                labels=np.concatenate(label_parts),
                image_codes=np.concatenate(image_parts),
                cycle_codes=np.concatenate(cycle_parts),
                batch_codes=np.concatenate(batch_parts),
                relative_codes=np.concatenate(relative_parts),
                image_paths=image_paths,
                image_labels=np.concatenate(image_label_parts),
            )
        )
    return timelines


def leave_one_tool_out_folds(tools: Sequence[str]) -> Iterator[ToolFold]:
    ordered = sorted(set(tools), key=lambda item: int(item[1:]))
    if len(ordered) < 3:
        raise ValueError("leave-one-tool-out with a separate validation tool requires at least 3 tools")
    for index, test_tool in enumerate(ordered):
        validation_tool = ordered[(index + 1) % len(ordered)]
        train_tools = tuple(tool for tool in ordered if tool not in {test_tool, validation_tool})
        yield ToolFold(train_tools, (validation_tool,), (test_tool,))


@dataclass(frozen=True)
class ForceNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, timelines: Sequence[ToolTimeline]) -> "ForceNormalizer":
        if not timelines:
            raise ValueError("cannot fit force normalization without training timelines")
        count = 0
        mean = np.zeros(3, dtype=np.float64)
        m2 = np.zeros(3, dtype=np.float64)
        for timeline in timelines:
            values = timeline.force.astype(np.float64, copy=False)
            chunk_mean = values.mean(axis=0)
            chunk_m2 = ((values - chunk_mean) ** 2).sum(axis=0)
            chunk_count = len(values)
            delta = chunk_mean - mean
            total = count + chunk_count
            mean += delta * chunk_count / total
            m2 += chunk_m2 + delta**2 * count * chunk_count / total
            count = total
        scale = np.sqrt(m2 / max(count, 1))
        scale[scale < 1e-8] = 1.0
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


class WindowDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        timelines: Sequence[ToolTimeline],
        normalizer: ForceNormalizer,
        input_len: int = 96,
        horizon: int = 24,
        stride: int = 24,
    ) -> None:
        if input_len < 1 or horizon < 1 or stride < 1:
            raise ValueError("window lengths and stride must be positive")
        self.timelines = list(timelines)
        self.normalizer = normalizer
        self.input_len = input_len
        self.horizon = horizon
        self.stride = stride
        self.offsets: list[tuple[int, int]] = []
        total = 0
        for timeline in self.timelines:
            count = max((len(timeline.force) - input_len - horizon) // stride + 1, 0)
            self.offsets.append((total, total + count))
            total += count
        self.total = total
        if total == 0:
            raise ValueError("no windows fit the selected input length and forecast horizon")

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += self.total
        if not 0 <= index < self.total:
            raise IndexError(index)
        for timeline_index, (start, end) in enumerate(self.offsets):
            if start <= index < end:
                timeline = self.timelines[timeline_index]
                window_start = (index - start) * self.stride
                break
        else:
            raise IndexError(index)
        input_end = window_start + self.input_len
        target_end = input_end + self.horizon
        return {
            "force": self.normalizer.transform(timeline.force[window_start:input_end]).copy(),
            "force_target": self.normalizer.transform(timeline.force[input_end:target_end]).copy(),
            "class_target": timeline.labels[input_end:target_end].astype(np.int64),
            "shown_label": np.int64(
                (
                    timeline.image_labels
                    if timeline.image_labels is not None
                    else timeline.labels
                )[input_end - 1]
            ),
            "image_paths": tuple(
                timeline.image_paths[code]
                for code in timeline.image_codes[window_start:input_end]
            ),
            "cycle": timeline.cycle_codes[window_start:input_end].astype(np.int64),
            "batch": timeline.batch_codes[window_start:input_end].astype(np.int64),
            "relative": timeline.relative_codes[window_start:input_end].astype(np.int64),
        }


def image_tensor(path: Path, height: int = 128, width: int = 384) -> Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGE_MEAN) / IMAGE_STD
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class ImageBank:
    """Cache resized RGB pixels once to avoid decoding the same photos for every batch."""

    def __init__(
        self, paths: Sequence[Path], height: int = 128, width: int = 384
    ) -> None:
        unique = list(dict.fromkeys(Path(path) for path in paths))
        self.height, self.width = height, width
        self.index = {path: position for position, path in enumerate(unique)}
        self.pixels = np.empty((len(unique), height, width, 3), dtype=np.uint8)
        for position, path in enumerate(unique):
            with Image.open(path) as source:
                resized = source.convert("RGB").resize(
                    (width, height), Image.Resampling.BILINEAR
                )
                self.pixels[position] = np.asarray(resized, dtype=np.uint8)

    @classmethod
    def from_timelines(
        cls, timelines: Sequence[ToolTimeline], height: int = 128, width: int = 384
    ) -> "ImageBank":
        return cls(
            [path for timeline in timelines for path in timeline.image_paths],
            height,
            width,
        )

    def __len__(self) -> int:
        return len(self.index)

    def tensor(self, path: Path) -> Tensor:
        array = self.pixels[self.index[Path(path)]].astype(np.float32) / 255.0
        array = (array - IMAGE_MEAN) / IMAGE_STD
        return torch.from_numpy(array.transpose(2, 0, 1).copy())


def collate_windows(
    samples: Sequence[dict[str, object]],
    height: int = 128,
    width: int = 384,
    bank: ImageBank | None = None,
) -> dict[str, object]:
    image_paths: list[list[Path]] = []
    image_indices: list[list[int]] = []
    for sample in samples:
        paths: list[Path] = []
        lookup: dict[Path, int] = {}
        row: list[int] = []
        for path in sample["image_paths"]:  # type: ignore[union-attr]
            path = Path(path)
            if path not in lookup:
                lookup[path] = len(paths)
                paths.append(path)
            row.append(lookup[path])
        image_paths.append(paths)
        image_indices.append(row)
    max_images = max(map(len, image_paths))
    load = bank.tensor if bank is not None else (
        lambda path: image_tensor(path, height, width)
    )
    images = [
        torch.stack(
            [load(path) for path in paths]
            + [load(paths[0])] * (max_images - len(paths))
        )
        for paths in image_paths
    ]
    return {
        "force": torch.from_numpy(np.stack([sample["force"] for sample in samples])),
        "force_target": torch.from_numpy(np.stack([sample["force_target"] for sample in samples])),
        "class_target": torch.from_numpy(np.stack([sample["class_target"] for sample in samples])),
        "shown_label": torch.from_numpy(np.stack([sample["shown_label"] for sample in samples])),
        "cycle": torch.from_numpy(np.stack([sample["cycle"] for sample in samples])),
        "batch": torch.from_numpy(np.stack([sample["batch"] for sample in samples])),
        "relative": torch.from_numpy(np.stack([sample["relative"] for sample in samples])),
        "images": torch.stack(images),
        "image_indices": torch.tensor(image_indices, dtype=torch.long),
    }
