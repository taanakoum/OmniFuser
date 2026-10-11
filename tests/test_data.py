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
    assert np.allclose(sample["force"][:6], sample["force"][0])
    assert np.allclose(sample["force"][6:], sample["force"][-1])
    assert sample["image_paths"][:6] == (first.image_path,) * 6
    assert sample["image_paths"][6:] == (second.image_path,) * 6
    assert np.array_equal(sample["class_target"], np.ones(6, dtype=np.int64))


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
