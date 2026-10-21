import json
from pathlib import Path

from omnifuser.cross_validation import _load_finished_fold
from omnifuser.data import nonastreda_source_signature
from omnifuser.train import _accumulation_group_sample_count


def test_partial_gradient_accumulation_group_is_sample_weighted() -> None:
    assert _accumulation_group_sample_count(5, 2, 1, 2) == 4
    assert _accumulation_group_sample_count(5, 2, 2, 2) == 4
    assert _accumulation_group_sample_count(5, 2, 3, 2) == 1


def test_finished_fold_reuse_checks_training_settings_and_artifacts(tmp_path: Path) -> None:
    artifacts = {
        "checkpoint": "fold-T1.pt",
        "test_predictions": "fold-T1-predictions.npz",
        "roc_curves": "fold-T1-roc.npz",
        "sensor_missing_predictions": "fold-T1-predictions-sensor_missing.npz",
        "sensor_missing_roc_curves": "fold-T1-roc-sensor_missing.npz",
        "image_missing_predictions": "fold-T1-predictions-image_missing.npz",
        "image_missing_roc_curves": "fold-T1-roc-image_missing.npz",
    }
    for name in artifacts.values():
        (tmp_path / name).write_bytes(b"artifact")
    config = {"prediction_head": "direct"}
    result = {
        "config": config,
        "source_signature": {"root": "dataset"},
        "pool": 10,
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "accumulation_steps": 2,
            "num_workers": 0,
            "seed": 7,
            "device": "cpu",
        },
        "artifacts": artifacts,
        "modality_evaluations": {
            mode: {"train": {}, "validation": {}, "test": {}}
            for mode in ("full", "sensor_missing", "image_missing")
        },
    }
    (tmp_path / "fold-T1.json").write_text(json.dumps(result), encoding="utf-8")
    expected = {
        "source_signature": {"root": "dataset"},
        "pool": 10,
        "training.learning_rate": 0.001,
        "training.accumulation_steps": 2,
        "training.seed": 7,
        "training.device": "cpu",
    }
    assert _load_finished_fold(tmp_path, "T1", config, expected, 2, 4) == result
    assert (
        _load_finished_fold(
            tmp_path,
            "T1",
            config,
            {**expected, "training.learning_rate": 0.01},
            2,
            4,
        )
        is None
    )
    # A fold written before missing-modality evaluation existed is retrained.
    legacy = {
        key: value for key, value in result.items() if key != "modality_evaluations"
    }
    (tmp_path / "fold-T1.json").write_text(json.dumps(legacy), encoding="utf-8")
    assert _load_finished_fold(tmp_path, "T1", config, expected, 2, 4) is None


def test_source_signature_tracks_dataset_files(tmp_path: Path) -> None:
    (tmp_path / "tool").mkdir()
    (tmp_path / "forces_xyz_raw.mat").write_bytes(b"force")
    (tmp_path / "labels.csv").write_text("labels", encoding="utf-8")
    (tmp_path / "labels_reg.csv").write_text("regression", encoding="utf-8")
    image_path = tmp_path / "tool" / "T1R1B1.jpg"
    image_path.write_bytes(b"image")
    original = nonastreda_source_signature(tmp_path)
    image_path.write_bytes(b"changed image")
    updated = nonastreda_source_signature(tmp_path)
    assert original != updated
