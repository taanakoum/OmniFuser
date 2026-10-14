import csv
import json

import pytest

from omnifuser.tracking import ExperimentTracker


def test_experiment_tracker_writes_metadata_jsonl_and_scalar_csv(tmp_path) -> None:
    tracker = ExperimentTracker(
        tmp_path,
        {"dataset": "Nonastreda", "model": {"width": 32}},
    )
    tracker.log(
        split="validation",
        fold="T10",
        epoch=2,
        metrics={
            "loss": 1.5,
            "classification": {"accuracy": 0.8, "auc": None},
            "per_horizon": [{"mse": 0.4}, {"mse": 0.7}],
        },
        context={"windows": 12, "evaluation_mode": "image_missing"},
    )

    metadata = json.loads((tmp_path / "tracking" / "run.json").read_text())
    with (tmp_path / "tracking" / "metrics.jsonl").open(encoding="utf-8") as file:
        event = json.loads(file.readline())
    with (tmp_path / "tracking" / "metrics.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    assert metadata["run_id"] == tracker.run_id
    assert event["split"] == "validation"
    assert event["context"]["evaluation_mode"] == "image_missing"
    assert event["metrics"]["classification"]["accuracy"] == 0.8
    assert {(row["metric"], float(row["value"])) for row in rows} == {
        ("loss", 1.5),
        ("classification.accuracy", 0.8),
        ("per_horizon.0.mse", 0.4),
        ("per_horizon.1.mse", 0.7),
    }
    assert {row["evaluation_mode"] for row in rows} == {"image_missing"}


def test_experiment_tracker_fails_on_corrupt_existing_manifest(tmp_path) -> None:
    tracking_dir = tmp_path / "tracking"
    tracking_dir.mkdir()
    (tracking_dir / "run.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid experiment tracking manifest"):
        ExperimentTracker(tmp_path, {"dataset": "Nonastreda"})


def test_experiment_tracker_rejects_mixed_run_metadata(tmp_path) -> None:
    ExperimentTracker(tmp_path, {"dataset": "Nonastreda", "seed": 1})

    with pytest.raises(ValueError, match="different configuration"):
        ExperimentTracker(tmp_path, {"dataset": "Nonastreda", "seed": 2})


def test_tracker_resumes_on_different_hardware_and_records_it(tmp_path) -> None:
    first = ExperimentTracker(
        tmp_path, {"seed": 1, "runtime": {"gpu_name": "NVIDIA T4", "torch": "2.5"}}
    )
    resumed = ExperimentTracker(
        tmp_path, {"seed": 1, "runtime": {"gpu_name": "NVIDIA L4", "torch": "2.6"}}
    )
    ExperimentTracker(
        tmp_path, {"seed": 1, "runtime": {"gpu_name": "NVIDIA L4", "torch": "2.6"}}
    )

    manifest = json.loads((tmp_path / "tracking" / "run.json").read_text())
    assert resumed.run_id == first.run_id
    assert manifest["metadata"]["runtime"]["gpu_name"] == "NVIDIA T4"
    assert [entry["runtime"]["gpu_name"] for entry in manifest["runtime_history"]] == [
        "NVIDIA L4"
    ]
    with pytest.raises(ValueError, match="different configuration"):
        ExperimentTracker(tmp_path, {"seed": 2, "runtime": {"gpu_name": "NVIDIA L4"}})


def test_tracker_migrates_legacy_csv_without_losing_metric_rows(tmp_path) -> None:
    tracker = ExperimentTracker(tmp_path, {"dataset": "Nonastreda"})
    csv_path = tmp_path / "tracking" / "metrics.csv"
    csv_path.write_text(
        "timestamp_utc,run_id,fold,epoch,split,metric,value\n"
        "now,run,T1,1,train,loss,2.5\n",
        encoding="utf-8",
    )

    resumed = ExperimentTracker(tmp_path, {"dataset": "Nonastreda"})
    resumed.log(
        split="test",
        fold="T1",
        epoch=1,
        metrics={"loss": 1.5},
        context={"evaluation_mode": "sensor_missing"},
    )

    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["evaluation_mode"] == ""
    assert rows[0]["value"] == "2.5"
    assert rows[1]["evaluation_mode"] == "sensor_missing"
    assert resumed.run_id == tracker.run_id
