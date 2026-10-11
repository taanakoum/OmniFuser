from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from scipy.signal import decimate
from torch import Tensor
from torch.utils.data import Dataset


CLASS_NAMES = ("sharp", "used", "dulled")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
RECORD_ID = re.compile(r"^T(?P<tool>\d+)R(?P<run>\d+)B(?P<block>\d+)$")
IMAGE_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True)
class Record:
    record_id: str
    tool: str
    run: int
    block: int
    force: np.ndarray
    label: int
    image_path: Path


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


def load_nonastreda(
    data_dir: str | Path,
    *,
    label_column: str = "image_label",
    decimation: int = 1,
    dedup_identical_forces: bool = True,
) -> list[Record]:
    """Load force/label/image records from the supplied MATLAB 5 MAT file."""
    root = Path(data_dir)
    if label_column not in {"image_label", "tool_label"}:
        raise ValueError("label_column must be 'image_label' or 'tool_label'")
    if decimation < 1:
        raise ValueError("decimation must be at least 1")
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
        for record_id, raw_force in raw_forces.items():
            tool, run, block, _ = _decode_record_id(record_id)
            if block != 4:
                continue
            previous_id = f"{tool}R{run}B3"
            previous = raw_forces.get(previous_id)
            if (
                previous is None
                or previous.shape != raw_force.shape
                or not np.array_equal(previous, raw_force)
            ):
                continue
            if flank_wear[record_id] > flank_wear[previous_id]:
                discarded_duplicate_ids.add(previous_id)
            else:
                discarded_duplicate_ids.add(record_id)

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
        force = raw_force.astype(np.float32, copy=False)
        if decimation > 1:
            force = decimate(force, decimation, axis=1, zero_phase=True).astype(np.float32, copy=False)
        records.append(
            Record(
                record_id=record_id,
                tool=tool,
                run=run,
                block=block,
                force=np.ascontiguousarray(force.T),
                label=class_index,
                image_path=image_path,
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


def make_tool_timelines(records: Sequence[Record]) -> list[ToolTimeline]:
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
        for record in selected:
            length = len(record.force)
            force_parts.append(record.force)
            label_parts.append(np.full(length, record.label, dtype=np.int8))
            image_parts.append(np.full(length, image_code[record.image_path], dtype=np.int16))
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
        for timeline_index, timeline in enumerate(self.timelines):
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


def collate_windows(
    samples: Sequence[dict[str, object]], height: int = 128, width: int = 384
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
    images = [
        torch.stack(
            [image_tensor(path, height, width) for path in paths]
            + [image_tensor(paths[0], height, width)] * (max_images - len(paths))
        )
        for paths in image_paths
    ]
    return {
        "force": torch.from_numpy(np.stack([sample["force"] for sample in samples])),
        "force_target": torch.from_numpy(np.stack([sample["force_target"] for sample in samples])),
        "class_target": torch.from_numpy(np.stack([sample["class_target"] for sample in samples])),
        "cycle": torch.from_numpy(np.stack([sample["cycle"] for sample in samples])),
        "batch": torch.from_numpy(np.stack([sample["batch"] for sample in samples])),
        "relative": torch.from_numpy(np.stack([sample["relative"] for sample in samples])),
        "images": torch.stack(images),
        "image_indices": torch.tensor(image_indices, dtype=torch.long),
    }
