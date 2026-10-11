import importlib.util
import json
from pathlib import Path

import numpy as np

from omnifuser.cross_validation import _load_finished_fold
from omnifuser.data import Record, nonastreda_source_signature
from omnifuser.train import _accumulation_group_sample_count


def test_partial_gradient_accumulation_group_is_sample_weighted() -> None:
    assert _accumulation_group_sample_count(5, 2, 1, 2) == 4
    assert _accumulation_group_sample_count(5, 2, 2, 2) == 4
    assert _accumulation_group_sample_count(5, 2, 3, 2) == 1


def test_diagnostic_normalizer_uses_pooled_records() -> None:
    module_spec = importlib.util.spec_from_file_location(
        "task_diagnostics",
        Path(__file__).parents[1] / "scripts" / "task_diagnostics.py",
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    first = Record(
        "T1R1B1",
        "T1",
        1,
        1,
        np.asarray([[0.0], [10.0], [0.0], [10.0]], dtype=np.float32),
        0,
        Path("one.jpg"),
    )
    pooled = module.pool_records([first], 2)
    mean, scale = module.fit_diagnostic_normalizer(pooled)
    assert np.allclose(mean, [5.0])
    assert np.allclose(scale, [1.0])
    assert not np.allclose(scale, first.force.std(axis=0))


def test_finished_fold_reuse_checks_training_settings_and_artifacts(tmp_path: Path) -> None:
    for name in ("fold-T1.pt", "fold-T1-predictions.npz", "fold-T1-roc.npz"):
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
        "artifacts": {
            "checkpoint": "fold-T1.pt",
            "test_predictions": "fold-T1-predictions.npz",
            "roc_curves": "fold-T1-roc.npz",
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
