"""Decision rules that turn per-lead class probabilities into wear classes."""
from __future__ import annotations

from itertools import pairwise, product

import numpy as np

from omnifuser.decision import (
    apply_offsets,
    argmax_decode,
    decision_scores,
    fit_class_offsets,
    monotone_decode,
    monotone_ordinal_decode,
    ordinal_median,
)


def _probabilities(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=float)[None]  # one window, (1, leads, 3)


def test_monotone_decode_is_the_best_non_decreasing_sequence() -> None:
    rng = np.random.default_rng(0)
    probabilities = rng.dirichlet(np.ones(3), size=(20, 6))
    decoded = monotone_decode(probabilities)
    assert (np.diff(decoded, axis=1) >= 0).all()
    log_p = np.log(probabilities)
    sequences = [s for s in product(range(3), repeat=6) if all(a <= b for a, b in pairwise(s))]
    for window in range(20):
        best = max(sequences, key=lambda s: sum(log_p[window, t, c] for t, c in enumerate(s)))
        assert tuple(decoded[window]) == best


def test_monotone_decode_removes_a_flicker_and_keeps_a_real_change() -> None:
    flicker = _probabilities([[0.2, 0.7, 0.1], [0.55, 0.4, 0.05], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6], [0.1, 0.2, 0.7]])
    assert argmax_decode(flicker).tolist() == [[1, 0, 1, 2, 2]]
    assert monotone_decode(flicker).tolist() == [[1, 1, 1, 2, 2]]


def test_monotone_ordinal_decode_is_the_best_non_decreasing_sequence_by_class_distance() -> None:
    rng = np.random.default_rng(3)
    probabilities = rng.dirichlet(np.ones(3), size=(15, 5))
    decoded = monotone_ordinal_decode(probabilities)
    distance = probabilities @ np.abs(np.arange(3)[:, None] - np.arange(3)[None, :])
    sequences = [s for s in product(range(3), repeat=5) if all(a <= b for a, b in pairwise(s))]
    for window in range(15):
        best = min(sequences, key=lambda s: sum(distance[window, t, c] for t, c in enumerate(s)))
        assert np.isclose(sum(distance[window, t, c] for t, c in enumerate(decoded[window])),
                          sum(distance[window, t, c] for t, c in enumerate(best)))
    # without a decrease to remove, it equals the per-lead ordinal median
    rising = _probabilities([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.1, 0.3, 0.6]])
    assert monotone_ordinal_decode(rising).tolist() == ordinal_median(rising).tolist() == [[0, 1, 2]]


def test_ordinal_median_picks_the_middle_class_when_mass_is_split_at_the_ends() -> None:
    probabilities = _probabilities([[0.45, 0.1, 0.45], [0.6, 0.3, 0.1], [0.1, 0.35, 0.55], [0.3, 0.3, 0.4]])
    assert argmax_decode(probabilities).tolist() == [[0, 0, 2, 2]]
    assert ordinal_median(probabilities).tolist() == [[1, 0, 2, 1]]


def test_zero_offsets_change_nothing_and_offsets_shift_decisions() -> None:
    rng = np.random.default_rng(1)
    probabilities = rng.dirichlet(np.ones(3), size=(5, 4))
    assert np.allclose(apply_offsets(probabilities, np.zeros(3)), probabilities)
    shifted = apply_offsets(probabilities, np.array([0.0, 5.0, 0.0]))
    assert (argmax_decode(shifted) == 1).mean() > (argmax_decode(probabilities) == 1).mean()


def test_fitted_offsets_correct_a_biased_boundary_and_keep_zero_without_gain() -> None:
    rng = np.random.default_rng(2)
    targets = rng.integers(0, 3, size=(40, 8))
    # a model that under-scores "used": the true class gets the mass, but used is scaled down
    probabilities = np.full((40, 8, 3), 0.2)
    np.put_along_axis(probabilities, targets[..., None], 0.6, axis=-1)
    probabilities[..., 1] *= 0.3
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    assert decision_scores(argmax_decode(probabilities), targets)["recall_class_1"] == 0.0
    offsets = fit_class_offsets(probabilities, targets)
    assert offsets[0] == 0.0 and offsets[1] > np.log(0.2 / 0.18)  # just enough to lift "used"
    assert decision_scores(argmax_decode(apply_offsets(probabilities, offsets)), targets)["accuracy"] == 1.0
    perfect = np.eye(3)[targets]
    assert np.allclose(fit_class_offsets(perfect, targets), 0.0)


def test_decision_scores_counts_changing_and_decreasing_windows() -> None:
    predicted = np.array([[0, 0, 1], [1, 1, 1], [2, 1, 1]])
    targets = np.array([[0, 0, 1], [1, 1, 1], [1, 1, 1]])
    scores = decision_scores(predicted, targets)
    assert scores["changing_windows"] == 2 / 3 and scores["decreasing_windows"] == 1 / 3
    assert scores["accuracy"] == 8 / 9 and scores["mean_absolute_class_error"] == 1 / 9
