"""Turning per-lead class probabilities into wear-class decisions.

The model outputs, for every window and forecast lead, probabilities over the ordered
classes sharp < used < dulled. ``argmax`` per lead is the default decision. The rules
here use two properties of tool wear that argmax ignores:

* the classes are ordered, so a decision can use cumulative probabilities
  (``ordinal_median``);
* wear does not decrease, so within one forecast window the class sequence should be
  non-decreasing (``monotone_decode``: the most probable non-decreasing sequence).

``fit_class_offsets`` tunes additive log-probability offsets on validation predictions
(never on test predictions), to correct biased decision boundaries.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np

EPSILON = 1e-9


def argmax_decode(probabilities: np.ndarray) -> np.ndarray:
    """(N, P, C) probabilities -> (N, P) classes, the most probable class per lead."""
    return probabilities.argmax(axis=-1)


def ordinal_median(probabilities: np.ndarray) -> np.ndarray:
    """The median class of each lead's ordinal distribution.

    Class ``k`` is chosen when P(class >= k) > 0.5 and P(class >= k + 1) <= 0.5. Unlike
    argmax, a lead split between two neighbouring classes cannot jump past the class in
    between, and a lead split between sharp and dulled is decided as used.
    """
    exceedance = np.flip(np.cumsum(np.flip(probabilities, axis=-1), axis=-1), axis=-1)
    return (exceedance[..., 1:] > 0.5).sum(axis=-1)


def monotone_decode(probabilities: np.ndarray) -> np.ndarray:
    """The most probable non-decreasing class sequence of each window (Viterbi).

    Maximizes the sum over leads of log p(class) subject to class[t] >= class[t - 1].
    """
    return _best_non_decreasing(np.log(np.clip(probabilities, EPSILON, 1.0)))


def monotone_ordinal_decode(probabilities: np.ndarray) -> np.ndarray:
    """The non-decreasing class sequence with the smallest expected class distance.

    Each lead costs sum_k p(k) |k - c|, whose per-lead minimizer is ``ordinal_median``;
    the sequence is constrained like ``monotone_decode``. This combines both rules
    without locking in an early error, as a running maximum of the median would.
    """
    classes = np.arange(probabilities.shape[-1])
    expected_distance = probabilities @ np.abs(classes[:, None] - classes[None, :]).astype(float)
    return _best_non_decreasing(-expected_distance)


def _best_non_decreasing(log_p: np.ndarray) -> np.ndarray:
    """argmax over non-decreasing sequences of the summed (N, P, C) per-lead scores."""
    windows, leads, classes = log_p.shape
    score = log_p[:, 0].copy()
    back = np.zeros((windows, leads, classes), dtype=np.int64)
    for lead in range(1, leads):
        # best predecessor among classes <= c: a running maximum over the class axis
        best_index = np.zeros((windows, classes), dtype=np.int64)
        best_score = np.empty((windows, classes))
        best_score[:, 0] = score[:, 0]
        for c in range(1, classes):
            better = score[:, c] > best_score[:, c - 1]
            best_score[:, c] = np.where(better, score[:, c], best_score[:, c - 1])
            best_index[:, c] = np.where(better, c, best_index[:, c - 1])
        back[:, lead] = best_index
        score = best_score + log_p[:, lead]
    path = np.empty((windows, leads), dtype=np.int64)
    path[:, -1] = score.argmax(axis=-1)
    rows = np.arange(windows)
    for lead in range(leads - 1, 0, -1):
        path[:, lead - 1] = back[rows, lead, path[:, lead]]
    return path


def apply_offsets(probabilities: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Probabilities after adding ``offsets`` (one per class) to their logarithms."""
    logits = np.log(np.clip(probabilities, EPSILON, 1.0)) + np.asarray(offsets)
    logits -= logits.max(axis=-1, keepdims=True)
    shifted = np.exp(logits)
    return shifted / shifted.sum(axis=-1, keepdims=True)


def decision_scores(predicted: np.ndarray, targets: np.ndarray, num_classes: int = 3) -> dict[str, float]:
    """Accuracy, macro precision/recall/F1, per-class recall, mean absolute class error,
    and the share of windows whose decided class changes within the forecast."""
    flat_p, flat_t = predicted.reshape(-1), targets.reshape(-1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (flat_t, flat_p), 1)
    true_positive = np.diag(confusion).astype(float)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, EPSILON)
    scores = {
        "accuracy": float(true_positive.sum() / max(confusion.sum(), 1)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "mean_absolute_class_error": float(np.abs(flat_p - flat_t).mean()),
        "changing_windows": float((predicted.max(axis=1) != predicted.min(axis=1)).mean()),
        "decreasing_windows": float((np.diff(predicted, axis=1) < 0).any(axis=1).mean()),
    }
    for index in range(num_classes):
        scores[f"recall_class_{index}"] = float(recall[index])
    return scores


def fit_class_offsets(
    probabilities: np.ndarray,
    targets: np.ndarray,
    decode: Callable[[np.ndarray], np.ndarray] = argmax_decode,
    objective: str = "f1_macro",
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Log-probability offsets (the first class fixed at 0) maximizing ``objective``.

    Fit on **validation** predictions only. Ties keep the smallest offsets, so with no
    gain the result is all zeros, which leaves decisions unchanged.
    """
    values = np.round(np.arange(-1.5, 1.51, 0.1), 2) if grid is None else np.asarray(grid)
    classes = probabilities.shape[-1]
    best = np.zeros(classes)
    best_score = decision_scores(decode(probabilities), targets, classes)[objective]
    candidates = sorted(product(values, repeat=classes - 1), key=lambda o: sum(abs(v) for v in o))
    for candidate in candidates:
        offsets = np.array((0.0, *candidate))
        score = decision_scores(decode(apply_offsets(probabilities, offsets)), targets, classes)[objective]
        if score > best_score + 1e-12:
            best, best_score = offsets, score
    return best
