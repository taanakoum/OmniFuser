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
        context={"windows": 12},
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
    assert event["metrics"]["classification"]["accuracy"] == 0.8
    assert {(row["metric"], float(row["value"])) for row in rows} == {
        ("loss", 1.5),
        ("classification.accuracy", 0.8),
        ("per_horizon.0.mse", 0.4),
        ("per_horizon.1.mse", 0.7),
    }


def test_experiment_tracker_fails_on_corrupt_existing_manifest(tmp_path) -> None:
    tracking_dir = tmp_path / "tracking"
    tracking_dir.mkdir()
    (tracking_dir / "run.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid experiment tracking manifest"):
        ExperimentTracker(tmp_path, {"dataset": "Nonastreda"})
