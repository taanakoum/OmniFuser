from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat

from omnifuser.data import (
    ForceNormalizer,
    Record,
    WindowDataset,
    collate_windows,
    load_nonastreda,
    leave_one_tool_out_folds,
    make_tool_timelines,
)


def _record(record_id: str, tool: str, run: int, block: int, value: float, label: int) -> Record:
    return Record(
        record_id=record_id,
        tool=tool,
        run=run,
        block=block,
        force=np.full((60, 3), value, dtype=np.float32),
        label=label,
        image_path=Path(f"{record_id}.jpg"),
    )


def test_tool_timeline_and_window_shapes() -> None:
    timeline = make_tool_timelines(
        [
            _record("T1R1B1", "T1", 1, 1, 1.0, 0),
            _record("T1R1B2", "T1", 1, 2, 2.0, 1),
        ]
    )[0]
    normalizer = ForceNormalizer.fit([timeline])
    dataset = WindowDataset([timeline], normalizer, input_len=12, horizon=6, stride=6)

    sample = dataset[0]
    assert sample["force"].shape == (12, 3)
    assert sample["force_target"].shape == (6, 3)
    assert sample["class_target"].shape == (6,)
    assert len(sample["image_paths"]) == 12
    assert len(dataset) == (120 - 12 - 6) // 6 + 1
    assert set(sample["class_target"].tolist()) == {0}


def test_windows_keep_force_label_and_image_alignment_across_records() -> None:
    first = _record("T1R1B1", "T1", 1, 1, 1.0, 0)
    second = _record("T1R1B2", "T1", 1, 2, 2.0, 1)
    timeline = make_tool_timelines([first, second])[0]
    dataset = WindowDataset(
        [timeline], ForceNormalizer.fit([timeline]), input_len=12, horizon=6, stride=6
    )

    sample = dataset[9]
    normalizer = dataset.normalizer
    assert np.allclose(sample["force"][:6], normalizer.transform(first.force[0]))
    assert np.allclose(sample["force"][6:], normalizer.transform(second.force[0]))
    assert not np.allclose(sample["force"][0], sample["force"][-1])
    assert sample["image_paths"][:6] == (first.image_path,) * 6
    assert sample["image_paths"][6:] == (second.image_path,) * 6
    assert np.array_equal(sample["class_target"], np.ones(6, dtype=np.int64))


def test_lag_one_uses_previous_pass_image_but_current_pass_target() -> None:
    first = _record("T1R1B1", "T1", 1, 1, 1.0, 0)
    second = _record("T1R1B2", "T1", 1, 2, 2.0, 1)
    timeline = make_tool_timelines([first, second], image_lag=1)[0]
    dataset = WindowDataset(
        [timeline], ForceNormalizer.fit([timeline]), input_len=12, horizon=6, stride=6
    )

    sample = dataset[9]

    assert sample["image_paths"] == (first.image_path,) * 12
    assert sample["shown_label"] == 0
    assert np.array_equal(sample["class_target"], np.ones(6, dtype=np.int64))


def _ramp_record(record_id: str, tool: str, block: int, start: float, length: int) -> Record:
    values = start + np.arange(length, dtype=np.float32)
    return Record(
        record_id=record_id,
        tool=tool,
        run=1,
        block=block,
        force=np.stack((values, -values, 2 * values), axis=1),
        label=block - 1,
        image_path=Path(f"{record_id}.jpg"),
    )


def test_forecast_target_is_exactly_the_next_horizon_samples() -> None:
    timelines = make_tool_timelines(
        [
            _ramp_record("T1R1B1", "T1", 1, 0.0, 50),
            _ramp_record("T1R1B2", "T1", 2, 50.0, 50),
            _ramp_record("T2R1B1", "T2", 1, 1000.0, 40),
        ]
    )
    normalizer = ForceNormalizer.fit(timelines)
    dataset = WindowDataset(timelines, normalizer, input_len=10, horizon=5, stride=7)

    # T1: (100 - 15) // 7 + 1 = 13 windows, T2: (40 - 15) // 7 + 1 = 4 windows
    assert len(dataset) == 17
    for index in range(len(dataset)):
        sample = dataset[index]
        timeline = timelines[0] if index < 13 else timelines[1]
        start = (index if index < 13 else index - 13) * 7
        raw_input = timeline.force[start : start + 10]
        raw_target = timeline.force[start + 10 : start + 15]
        assert np.allclose(sample["force"], normalizer.transform(raw_input))
        assert np.allclose(sample["force_target"], normalizer.transform(raw_target))
        # the target continues the input ramp, so no sample is skipped or reused
        restored = sample["force_target"] * normalizer.scale + normalizer.mean
        assert np.allclose(restored[0, 0], raw_input[-1, 0] + 1)
        assert np.array_equal(sample["class_target"], timeline.labels[start + 10 : start + 15])
    # windows never mix tools: T2 windows only see T2's 1000+ force values
    assert dataset[13]["force"][:, 0].min() * normalizer.scale[0] + normalizer.mean[0] >= 999.0


def test_label_switch_inside_target_follows_record_boundary() -> None:
    timeline = make_tool_timelines(
        [
            _ramp_record("T1R1B1", "T1", 1, 0.0, 20),
            _ramp_record("T1R1B2", "T1", 2, 20.0, 20),
        ]
    )[0]
    dataset = WindowDataset(
        [timeline], ForceNormalizer.fit([timeline]), input_len=16, horizon=8, stride=16
    )
    sample = dataset[0]
    # leads 1-4 are still record 1 (label 0), leads 5-8 are record 2 (label 1)
    assert sample["class_target"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert sample["shown_label"] == 0


def test_normalizer_pools_statistics_across_training_tools() -> None:
    rng = np.random.default_rng(3)
    records = [
        Record(f"T{tool}R1B1", f"T{tool}", 1, 1,
               rng.normal(tool * 10, tool, (100 + 37 * tool, 3)).astype(np.float32),
               0, Path(f"T{tool}.jpg"))
        for tool in (1, 2, 3)
    ]
    timelines = make_tool_timelines(records)
    normalizer = ForceNormalizer.fit(timelines)
    pooled = np.concatenate([timeline.force for timeline in timelines]).astype(np.float64)

    assert np.allclose(normalizer.mean, pooled.mean(axis=0), rtol=1e-5)
    assert np.allclose(normalizer.scale, pooled.std(axis=0), rtol=1e-5)
    transformed = np.concatenate(
        [normalizer.transform(timeline.force) for timeline in timelines]
    )
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-4)


def test_loto_keeps_each_tool_in_one_split() -> None:
    folds = list(leave_one_tool_out_folds(["T1", "T2", "T3", "T4"]))
    fold = folds[0]
    assert fold.test_tools == ("T1",)
    assert fold.validation_tools == ("T2",)
    assert fold.train_tools == ("T3", "T4")
    assert not (set(fold.train_tools) & set(fold.validation_tools))
    assert not (set(fold.train_tools) & set(fold.test_tools))


def test_every_loto_fold_partitions_tools_without_overlap() -> None:
    tools = [f"T{index}" for index in range(1, 11)]
    folds = list(leave_one_tool_out_folds(tools))
    assert len(folds) == len(tools)
    assert {fold.test_tools[0] for fold in folds} == set(tools)
    for fold in folds:
        assert len(fold.train_tools) == 8
        assert len(fold.validation_tools) == 1
        assert len(fold.test_tools) == 1
        assert set(fold.train_tools).isdisjoint(fold.validation_tools)
        assert set(fold.train_tools).isdisjoint(fold.test_tools)
        assert set(fold.validation_tools).isdisjoint(fold.test_tools)
        assert set(fold.train_tools) | set(fold.validation_tools) | set(fold.test_tools) == set(tools)


def test_normalizer_is_fitted_on_selected_training_timelines() -> None:
    training = make_tool_timelines([_record("T1R1B1", "T1", 1, 1, 3.0, 0)])
    held_out = make_tool_timelines([_record("T2R1B1", "T2", 1, 1, 100.0, 0)])
    normalizer = ForceNormalizer.fit(training)
    assert np.allclose(normalizer.mean, 3.0)
    assert np.allclose(normalizer.transform(held_out[0].force), 97.0)


def test_collate_keeps_images_batched_and_indices_local(tmp_path: Path) -> None:
    paths = [tmp_path / f"tool-{index}.jpg" for index in range(3)]
    for index, path in enumerate(paths):
        Image.new("RGB", (20, 10), color=(index * 80, 20, 40)).save(path)
    common = {
        "force": np.zeros((4, 3), dtype=np.float32),
        "force_target": np.zeros((2, 3), dtype=np.float32),
        "class_target": np.zeros(2, dtype=np.int64),
        "shown_label": np.int64(0),
        "cycle": np.zeros(4, dtype=np.int64),
        "batch": np.zeros(4, dtype=np.int64),
        "relative": np.zeros(4, dtype=np.int64),
    }
    samples = [
        {**common, "image_paths": (paths[0], paths[1], paths[0], paths[1])},
        {**common, "image_paths": (paths[2],) * 4},
    ]

    batch = collate_windows(samples, height=8, width=16)
    assert batch["images"].shape == (2, 2, 3, 8, 16)
    assert batch["image_indices"].shape == (2, 4)
    assert batch["image_indices"][0].tolist() == [0, 1, 0, 1]
    assert batch["image_indices"][1].tolist() == [0, 0, 0, 0]


def test_exact_force_duplicates_keep_most_worn_annotation_per_run(tmp_path: Path) -> None:
    (tmp_path / "tool").mkdir()
    rows = np.empty((4, 5), dtype=object)
    force = np.arange(3 * 120, dtype=np.float64).reshape(3, 120)
    duplicate_force = force + 1000
    ids = [f"T1R1B{block}" for block in range(1, 5)]
    labels = ("sharp", "sharp", "used", "dulled")
    for index, record_id in enumerate(ids):
        filename = f"{record_id}.jpg"
        Image.new("RGB", (20, 10)).save(tmp_path / "tool" / filename)
        rows[index] = [
            filename,
            labels[index],
            "1",
            force.copy() if index in {0, 1} else duplicate_force.copy(),
            "",
        ]
    savemat(tmp_path / "forces_xyz_raw.mat", {"baseDatastore": rows})
    with (tmp_path / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        file.write("id,image_label,tool_label\n")
        file.writelines(
            f"{record_id},{label},1\n" for record_id, label in zip(ids, labels)
        )
    with (tmp_path / "labels_reg.csv").open("w", newline="", encoding="utf-8") as file:
        file.write("id,gaps,flank_wear,overhang\n")
        file.writelines(
            f"{record_id},0,{wear},0\n"
            for record_id, wear in zip(ids, (10, 20, 30, 40))
        )

    kept = load_nonastreda(
        tmp_path, decimation=1, dedup_identical_forces=False, trim_pass=False
    )
    deduplicated = load_nonastreda(tmp_path, decimation=1, trim_pass=False)
    assert len(kept) == 4
    assert len(deduplicated) == 2
    assert {record.record_id for record in deduplicated} == {
        "T1R1B2",
        "T1R1B4",
    }
    selected_first_duplicate = next(r for r in deduplicated if r.block == 2)
    assert selected_first_duplicate.label == 0
    assert selected_first_duplicate.image_path.name == "T1R1B2.jpg"
    selected_duplicate = next(r for r in deduplicated if r.block == 4)
    assert selected_duplicate.label == 2
    assert selected_duplicate.image_path.name == "T1R1B4.jpg"


def test_default_span_trimming_preserves_record_label_and_image(tmp_path: Path) -> None:
    (tmp_path / "tool").mkdir()
    record_ids = ("T1R1B1", "T1R1B2")
    for record_id in record_ids:
        Image.new("RGB", (20, 10)).save(tmp_path / "tool" / f"{record_id}.jpg")
    force = np.zeros((3, 1000), dtype=np.float64)
    force[:, 10:990] = 10
    force[:, 500:510] = 0
    rows = np.empty((2, 5), dtype=object)
    rows[0] = [f"{record_ids[0]}.jpg", "used", "1", force, ""]
    rows[1] = [
        f"{record_ids[1]}.jpg",
        "dulled",
        "1",
        np.full((3, 1000), 20, dtype=np.float64),
        "",
    ]
    savemat(tmp_path / "forces_xyz_raw.mat", {"baseDatastore": rows})
    with (tmp_path / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        file.write("id,image_label,tool_label\n")
        file.writelines(f"{record_id},{label},1\n" for record_id, label in zip(record_ids, ("used", "dulled")))
    with (tmp_path / "labels_reg.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        file.write("id,gaps,flank_wear,overhang\n")
        file.writelines(f"{record_id},0,40,0\n" for record_id in record_ids)

    trimmed = load_nonastreda(tmp_path)[0]
    untrimmed = load_nonastreda(tmp_path, trim_pass=False)[0]

    assert len(untrimmed.force) == 1000
    assert 900 < len(trimmed.force) < len(untrimmed.force)
    assert trimmed.label == 1
    assert trimmed.image_path == tmp_path / "tool" / f"{record_ids[0]}.jpg"
    assert np.any(np.all(trimmed.force == 0, axis=1))
    assert trimmed.source_length_samples == 1000
    assert trimmed.trimmed_lead_samples > 0
    assert trimmed.trimmed_runout_samples > 0
    assert untrimmed.trimmed_lead_samples == 0
    assert untrimmed.trimmed_runout_samples == 0


def test_span_trim_keeps_a_lead_in_longer_than_half_the_reference_window() -> None:
    """Documents the rule's known limit; EDA.md lists the affected real passes."""
    from omnifuser.data import _nonastreda_span_bounds

    short = np.full((10_000, 3), 50.0)
    short[:200] = 0.5  # 2 % lead-in: the early window still measures cutting level
    assert 190 <= _nonastreda_span_bounds(short)[0] <= 210

    long = np.full((10_000, 3), 50.0)
    long[:600] = 0.5  # 6 % lead-in fills most of the 5 % reference window
    assert _nonastreda_span_bounds(long)[0] == 0
