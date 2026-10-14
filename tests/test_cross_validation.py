import numpy as np
import pytest

from omnifuser.cross_validation import (
    _aggregate_out_of_fold,
    _summarize_modality_evaluations,
    _summarize_folds,
    _summarize_paper_horizons,
)
from omnifuser.evaluation import classification_report


def test_cross_validation_aggregates_predictions_and_mean_roc(tmp_path) -> None:
    probabilities = np.asarray(
        [[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]],
        dtype=np.float32,
    )
    targets = np.asarray([[0, 1, 2]], dtype=np.int64)
    classification, curves = classification_report(probabilities, targets)
    np.savez_compressed(
        tmp_path / "predictions.npz",
        force_predictions=np.zeros((1, 3, 3), dtype=np.float32),
        force_targets=np.ones((1, 3, 3), dtype=np.float32),
    )
    np.savez_compressed(
        tmp_path / "roc.npz",
        **{
            f"{name}_{coordinate}": values
            for name, curve in curves.items()
            for coordinate, values in zip(("fpr", "tpr", "thresholds"), curve)
        },
    )
    result = {
        "test_tool": "T1",
        "artifacts": {
            "test_predictions": "predictions.npz",
            "roc_curves": "roc.npz",
        },
        "test": {
            "classification": classification,
            "regression": {"overall": {"mae": 1.0, "mse": 1.0, "rmse": 1.0}},
        },
    }

    aggregate = _aggregate_out_of_fold([result], tmp_path)

    assert aggregate["classification"]["accuracy"] == 1.0
    assert aggregate["classification"]["macro_mean_roc_auc"] == pytest.approx(1.0, abs=1e-3)
    assert aggregate["regression"]["overall"]["mae"] == 1.0
    assert (tmp_path / "cross-validation-mean-roc.npz").is_file()


def test_fold_summary_keeps_missing_auc_fold_slots() -> None:
    result = {
        "test": {
            "classification_accuracy": 0.5,
            "classification": {
                "precision_macro": 0.4,
                "recall_macro": 0.3,
                "f1_macro": 0.35,
                "roc_auc_ovr": {"macro": None},
            },
            "regression": {"overall": {"mae": 2.0, "mse": 4.0, "rmse": 2.0}},
        }
    }

    summary = _summarize_folds([result])

    assert summary["metrics"]["classification_roc_auc_macro"]["per_fold"] == [None]
    assert summary["metrics"]["classification_roc_auc_macro"]["mean"] is None


def test_fold_summary_includes_weighted_metrics_and_paper_horizons() -> None:
    result = {
        "test": {
            "classification": {
                "precision_weighted": 0.7,
                "recall_weighted": 0.8,
                "f1_weighted": 0.75,
            },
            "paper_horizons": {"12": {"mse": 0.5, "mae": 0.4}},
            "baselines": {
                "by_horizon": {"12": {"last_value_mse": 0.8}}
            },
        }
    }

    summary = _summarize_folds([result])
    horizon_summary = _summarize_paper_horizons([result])

    assert summary["metrics"]["classification_f1_weighted"]["mean"] == 0.75
    assert horizon_summary["models"]["12"]["mse"]["mean"] == 0.5
    assert horizon_summary["baselines"]["12"]["last_value_mse"]["mean"] == 0.8


def test_modality_summary_preserves_stages_and_missing_class_auc() -> None:
    metric = {
        "classification": {
            "accuracy": 0.7,
            "precision_macro": 0.6,
            "recall_macro": 0.5,
            "f1_macro": 0.55,
            "precision_weighted": 0.7,
            "recall_weighted": 0.7,
            "f1_weighted": 0.68,
            "roc_auc_ovr": {"macro": None},
        },
        "regression": {
            "overall": {"mae": 1.0, "mse": 2.0, "rmse": 2**0.5}
        },
    }
    result = {
        "modality_evaluations": {
            mode: {stage: metric for stage in ("train", "validation", "test")}
            for mode in ("full", "sensor_missing", "image_missing")
        }
    }

    summary = _summarize_modality_evaluations([result])

    assert summary["image_missing"]["validation"]["classification_accuracy"]["mean"] == 0.7
    assert summary["sensor_missing"]["test"]["classification_roc_auc_macro"]["mean"] is None
    assert summary["full"]["train"]["regression_mse"]["per_fold"] == [2.0]
