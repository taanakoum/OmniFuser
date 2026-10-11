import numpy as np
import pytest

from omnifuser.evaluation import classification_report, regression_report


def test_classification_report_includes_metrics_horizon_and_roc() -> None:
    probabilities = np.asarray(
        [
            [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]],
            [[0.2, 0.7, 0.1], [0.1, 0.1, 0.8]],
            [[0.8, 0.1, 0.1], [0.1, 0.9, 0.0]],
        ]
    )
    targets = np.asarray([[0, 1], [1, 2], [0, 1]])

    report, curves = classification_report(probabilities, targets)

    assert report["accuracy"] == pytest.approx(1.0)
    assert len(report["by_horizon"]) == 2
    assert report["per_class"]["sharp"]["support"] == 2
    assert report["roc_auc_ovr"]["per_class"]["sharp"] is not None
    assert set(curves) == {"sharp", "used", "dulled"}
    for fpr, tpr, thresholds in curves.values():
        assert fpr[0] == 0 and tpr[0] == 0
        assert fpr[-1] == 1 and tpr[-1] == 1
        assert len(fpr) == len(tpr) == len(thresholds)


def test_roc_auc_is_undefined_when_a_class_is_absent() -> None:
    probabilities = np.asarray([[[0.8, 0.2, 0.0], [0.7, 0.3, 0.0]]])
    targets = np.asarray([[0, 1]])
    report, curves = classification_report(probabilities, targets)

    assert report["roc_auc_ovr"]["per_class"]["dulled"] is None
    assert "dulled" not in curves


def test_regression_report_is_in_original_force_units() -> None:
    prediction = np.zeros((1, 2, 3), dtype=np.float32)
    target = np.ones_like(prediction)
    mean = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    scale = np.asarray([2.0, 3.0, 4.0], dtype=np.float32)

    report = regression_report(prediction, target, mean, scale)

    assert report["units"] == "original force units"
    assert report["overall"]["mae"] == pytest.approx(3.0)
    assert report["overall"]["mse"] == pytest.approx(29 / 3)
    assert report["by_channel"]["Fx"]["rmse"] == pytest.approx(2.0)
    assert len(report["by_horizon"]) == 2
