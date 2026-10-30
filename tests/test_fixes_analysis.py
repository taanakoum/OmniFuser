"""The fixes study's seed averaging, paired comparison and decision rules."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import scripts.fixes_analysis as analysis


def _write_run(path: Path, rng: np.random.Generator, mse: float, sharpen: float) -> None:
    path.mkdir(parents=True)
    (path / "cross-validation-results.json").write_text("{}", encoding="utf-8")
    for index in range(1, 11):
        targets = np.repeat(np.arange(3), 8).reshape(4, 6)
        noise = rng.dirichlet(np.ones(3), size=targets.shape)
        probabilities = (1 - sharpen) * noise + sharpen * np.eye(3)[targets]
        np.savez(path / f"fold-T{index}-predictions.npz", class_probabilities=probabilities, class_targets=targets)
        result = {
            "test": {"regression": {"normalized": {"mse": mse + 0.01 * rng.random()}}},
            "modality_evaluations": {"full": {"train": {"classification_accuracy": 0.9}},
                                     "image_missing": {"test": {"classification_accuracy": 0.5}}},
        }
        (path / f"fold-T{index}.json").write_text(json.dumps(result), encoding="utf-8")


def test_analysis_adopts_an_arm_that_meets_its_rule_and_rejects_one_that_does_not(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    for seed in ("s42", "s43"):
        _write_run(tmp_path / seed / "fixes" / "reference", rng, mse=0.80, sharpen=0.3)
        _write_run(tmp_path / seed / "fixes" / "revin", rng, mse=0.60, sharpen=0.3)  # same class quality
        _write_run(tmp_path / seed / "fixes" / "no-image-gat", rng, mse=0.80, sharpen=0.3)
    result = analysis.analyse(tmp_path)
    assert result["seeds"] == ["s42", "s43"]
    assert result["decisions"]["revin"]["adopted"]
    assert not result["decisions"]["no-image-gat"]["adopted"]
    assert result["stage2_flags"] == "--revin"
    reference = result["arms"]["reference"]
    assert len(reference["per_fold"]["accuracy"]) == 10
    assert abs(reference["mean"]["train_test_gap"] - (0.9 - reference["mean"]["accuracy"])) < 1e-9
    assert set(reference["seed_ensemble"]) == {"accuracy", "accuracy_ordinal_monotone", "f1_macro"}


def test_paired_comparison_counts_folds_and_skips_missing_values() -> None:
    test = analysis.paired([0.7, 0.8, None, 0.6], [0.6, 0.8, 0.5, 0.5])
    assert test is not None
    assert test["folds_up"] == 2 and test["folds_down"] == 0
    assert abs(test["difference"] - (0.1 + 0.0 + 0.1) / 3) < 1e-12
    assert analysis.paired([None, None], [0.1, 0.2]) is None
