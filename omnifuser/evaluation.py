from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from omnifuser.data import CLASS_NAMES


PAPER_HORIZONS = (12, 24, 36, 48, 60, 72, 84, 96)


def _macro_scores(targets: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    true_positive = np.diag(confusion)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    support = confusion.sum(axis=1) / max(confusion.sum(), 1)
    return {
        "accuracy": float(true_positive.sum() / max(confusion.sum(), 1)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "precision_weighted": float((support * precision).sum()),
        "recall_weighted": float((support * recall).sum()),
        "f1_weighted": float((support * f1).sum()),
    }


def cumulative_horizon_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_targets: np.ndarray,
    horizons: tuple[int, ...] = PAPER_HORIZONS,
) -> dict[str, dict[str, float]]:
    """Report metrics over leads 1..h for each paper horizon that fits the output."""
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("force predictions and targets must match (N, horizon, channels)")
    if probabilities.ndim != 3 or probabilities.shape[:2] != class_targets.shape:
        raise ValueError("class probabilities and targets must match (N, horizon, classes)")
    if probabilities.shape[:2] != predictions.shape[:2]:
        raise ValueError("force and classification horizons must match")
    report: dict[str, dict[str, float]] = {}
    for horizon in horizons:
        if horizon < 1:
            raise ValueError("reported horizons must be positive")
        if horizon > predictions.shape[1]:
            continue
        error = predictions[:, :horizon] - targets[:, :horizon]
        report[str(horizon)] = {
            "mse": float(np.square(error).mean()),
            "mae": float(np.abs(error).mean()),
            **_macro_scores(
                class_targets[:, :horizon].reshape(-1),
                probabilities[:, :horizon].argmax(axis=-1).reshape(-1),
            ),
        }
    return report


def baseline_report(
    last_input_force: np.ndarray,
    targets: np.ndarray,
    shown_label: np.ndarray,
    class_targets: np.ndarray,
    horizons: tuple[int, ...] = PAPER_HORIZONS,
) -> dict[str, Any]:
    """Simple normalized-force and shown-photo-label persistence baselines."""
    if targets.ndim != 3 or last_input_force.shape != (targets.shape[0], targets.shape[2]):
        raise ValueError("baseline force arrays have incompatible shapes")
    if class_targets.ndim != 2 or class_targets.shape != targets.shape[:2]:
        raise ValueError("baseline class targets must match (N, horizon)")
    if shown_label.shape != (targets.shape[0],):
        raise ValueError("shown labels must have shape (N,)")
    def score(upto: int) -> dict[str, float]:
        last_value_error = targets[:, :upto] - last_input_force[:, None, :]
        return {
            "last_value_mse": float(np.square(last_value_error).mean()),
            "last_value_mae": float(np.abs(last_value_error).mean()),
            "zero_prediction_mse": float(np.square(targets[:, :upto]).mean()),
            "persistence_class_accuracy": float(
                np.mean(class_targets[:, :upto] == shown_label[:, None])
            ),
        }

    report: dict[str, Any] = score(targets.shape[1])
    report["by_horizon"] = {
        str(horizon): score(horizon)
        for horizon in horizons
        if 1 <= horizon <= targets.shape[1]
    }
    return report


def _binary_roc_curve(
    targets: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    positives = targets.astype(bool)
    positive_count = int(positives.sum())
    negative_count = int(len(positives) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return np.array([]), np.array([]), np.array([]), None

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_positive = positives[order]
    distinct = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    true_positive = np.cumsum(sorted_positive)[distinct]
    false_positive = 1 + distinct - true_positive
    tpr = np.r_[0.0, true_positive / positive_count]
    fpr = np.r_[0.0, false_positive / negative_count]
    thresholds = np.r_[np.inf, sorted_scores[distinct]]
    auc = float(np.sum((tpr[1:] + tpr[:-1]) * np.diff(fpr) * 0.5))
    return fpr, tpr, thresholds, auc


def classification_report(
    probabilities: np.ndarray, targets: np.ndarray
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    if probabilities.ndim != 3 or targets.shape != probabilities.shape[:2]:
        raise ValueError("expected probabilities (N, horizon, classes) and targets (N, horizon)")
    if probabilities.shape[-1] != len(CLASS_NAMES):
        raise ValueError(f"expected {len(CLASS_NAMES)} class probabilities")
    if not np.isfinite(probabilities).all():
        raise ValueError("classification probabilities contain non-finite values")
    if not np.isin(targets, np.arange(len(CLASS_NAMES))).all():
        raise ValueError("classification targets contain an unknown class")

    predicted = probabilities.argmax(axis=-1)
    flat_targets = targets.reshape(-1)
    flat_predicted = predicted.reshape(-1)
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (flat_targets, flat_predicted), 1)
    true_positive = np.diag(confusion)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    support = confusion.sum(axis=1) / max(confusion.sum(), 1)
    report: dict[str, Any] = {
        "accuracy": float(np.mean(flat_targets == flat_predicted)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "precision_weighted": float((support * precision).sum()),
        "recall_weighted": float((support * recall).sum()),
        "f1_weighted": float((support * f1).sum()),
        "confusion_matrix": confusion.tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(confusion[index].sum()),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }

    horizon_metrics: list[dict[str, float]] = []
    for step in range(targets.shape[1]):
        step_targets = targets[:, step]
        step_predicted = predicted[:, step]
        step_confusion = np.zeros_like(confusion)
        np.add.at(step_confusion, (step_targets, step_predicted), 1)
        step_tp = np.diag(step_confusion)
        step_precision = step_tp / np.maximum(step_confusion.sum(axis=0), 1)
        step_recall = step_tp / np.maximum(step_confusion.sum(axis=1), 1)
        step_f1 = (
            2 * step_precision * step_recall
            / np.maximum(step_precision + step_recall, 1e-12)
        )
        step_support = step_confusion.sum(axis=1) / max(step_confusion.sum(), 1)
        horizon_metrics.append(
            {
                "accuracy": float(np.mean(step_targets == step_predicted)),
                "precision_macro": float(step_precision.mean()),
                "recall_macro": float(step_recall.mean()),
                "f1_macro": float(step_f1.mean()),
                "precision_weighted": float((step_support * step_precision).sum()),
                "recall_weighted": float((step_support * step_recall).sum()),
                "f1_weighted": float((step_support * step_f1).sum()),
            }
        )
    report["by_horizon"] = horizon_metrics

    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    aucs: dict[str, float | None] = {}
    flat_probabilities = probabilities.reshape(-1, probabilities.shape[-1])
    for index, name in enumerate(CLASS_NAMES):
        fpr, tpr, thresholds, auc = _binary_roc_curve(
            flat_targets == index, flat_probabilities[:, index]
        )
        aucs[name] = auc
        if auc is not None:
            curves[name] = (fpr, tpr, thresholds)
    valid_aucs = [value for value in aucs.values() if value is not None]
    report["roc_auc_ovr"] = {
        "per_class": aucs,
        "macro": float(np.mean(valid_aucs)) if valid_aucs else None,
    }
    return report, curves


def regression_report(
    predictions: np.ndarray, targets: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> dict[str, Any]:
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("expected matching regression arrays with shape (N, horizon, channels)")
    physical_predictions = predictions * scale[None, None, :] + mean[None, None, :]
    physical_targets = targets * scale[None, None, :] + mean[None, None, :]
    error = physical_predictions - physical_targets
    normalized_error = predictions - targets

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "mae": float(np.abs(values).mean()),
            "mse": float(np.square(values).mean()),
            "rmse": float(np.sqrt(np.square(values).mean())),
        }

    by_channel = {
        channel: summarize(error[..., index])
        for index, channel in enumerate(("Fx", "Fy", "Fz"))
    }
    by_horizon = [
        summarize(error[:, step, :]) for step in range(error.shape[1])
    ]
    by_horizon_normalized = [
        summarize(normalized_error[:, step, :])
        for step in range(normalized_error.shape[1])
    ]
    return {
        "units": "original force units",
        "overall": summarize(error),
        "by_channel": by_channel,
        "by_horizon": by_horizon,
        "by_horizon_normalized": by_horizon_normalized,
        "normalized": summarize(normalized_error),
    }


def softmax_probabilities(logits: Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=-1).detach().cpu().numpy()
