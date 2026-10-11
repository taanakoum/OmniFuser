"""NumPy reference for Blocks 9-11 of OmniFuser (arXiv 2511.01320v1): recursive refinement and the prediction heads
(Eq. 20-23), the training objective, and the scores of Sec. VI-A2 / Tables IV-V.

Written from the paper's text, independent of ``omnifuser/``.  Extends ``tests/c2f_reference.py`` (Blocks 6-8).
One sample, float64, sequences are (T, d).  [CHOICE] marks what the paper does not settle.

    Block 9   refine(), head_*()                        Eq. 20, 21-23
    Block 10  multitask_loss()                          Sec. VI-A3 (the loss itself is not stated; MSE + CE is the usual choice)
    Block 11  confusion(), class_scores(), roc_auc_ovr(), cumulative_*()   Sec. VI-A2, Tables IV and V
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import rankdata

import c2f_reference as c2f

# =================================================================================================== Block 9
def refine(
    z_r: np.ndarray,
    z_i: np.ndarray,
    layers: list[dict[str, Any]],
    proj_r: list[np.ndarray],
    proj_i: list[np.ndarray],
    **layer_options: Any,
) -> list[np.ndarray]:
    """Eq. 20.  Z^(1) = G(Z^r, Z^i);  Z^(k) = G(Z^r + P^r(Z^(k-1)), Z^i + P^i(Z^(k-1))),  k = 2..n.

    ``layers`` holds the parameters of G for each round (n entries).  The paper writes one symbol G for all rounds;
    Fig. 4 stacks Layer 1..n.  Passing the same dict n times is the weight-sharing reading.
    proj_r / proj_i hold the n-1 matrices of P^r, P^i as (d, d) maps applied from the right [CHOICE: linear, no bias].
    Returns [Z^(1), ..., Z^(n)].
    """
    fused = [c2f.c2f_layer(z_r, z_i, layers[0], **layer_options)["Z_hat"]]
    for k in range(1, len(layers)):
        previous = fused[-1]
        fused.append(c2f.c2f_layer(z_r + previous @ proj_r[k - 1], z_i + previous @ proj_i[k - 1], layers[k],
                                   **layer_options)["Z_hat"])
    return fused


def head_paper_literal(z_t: np.ndarray, w_reg, b_reg, w_cls, b_cls, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Eq. 21-22 as printed: y_t = W_r z_T + b_r and o_t = f_cls(z_T) do not depend on t, so every lead gets the same value."""
    y = z_t @ w_reg + b_reg
    o = z_t @ w_cls + b_cls
    return np.repeat(y[None], horizon, axis=0), np.repeat(o[None], horizon, axis=0)


def head_direct(z_t: np.ndarray, w_reg, b_reg, w_cls, b_cls, horizon: int, channels: int, classes: int = 3):
    """Eq. 21-22 with the dependence on t made explicit: one (W_r^(t), b_r^(t)) per lead.  Stored as d x (P C) and d x (P 3)."""
    return (z_t @ w_reg + b_reg).reshape(horizon, channels), (z_t @ w_cls + b_cls).reshape(horizon, classes)


def head_lead_conditioned(z_t: np.ndarray, lead_table, w_proj, b_proj, w_reg, b_reg, w_cls, b_cls):
    """[CHOICE of the repo] lead embedding e_t, h_t = gelu([z_T, e_t] W + b), y_t = W_r h_t + b_r, o_t = W_c h_t + b_c."""
    horizon = lead_table.shape[0]
    context = np.repeat(z_t[None], horizon, axis=0)
    h = c2f.gelu(np.concatenate([context, lead_table], axis=-1) @ w_proj + b_proj)
    return h @ w_reg + b_reg, h @ w_cls + b_cls


# =================================================================================================== Block 10
def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean over all (sample, lead) pairs of -log softmax(logits)[label].  logits (N, P, 3), labels (N, P)."""
    flat = logits.reshape(-1, logits.shape[-1])
    z = flat - flat.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-log_p[np.arange(len(flat)), labels.reshape(-1)].mean())


def multitask_loss(force_pred, force_true, logits, labels) -> dict[str, float]:
    """[CHOICE] L = MSE(force) + CE(class), weights 1 and 1.  The paper names both tasks but gives no loss or weights."""
    mse = float(np.mean((force_pred - force_true) ** 2))
    ce = softmax_cross_entropy(logits, labels)
    return {"force_loss": mse, "class_loss": ce, "loss": mse + ce}


# =================================================================================================== Block 11
def confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: int = 3) -> np.ndarray:
    cm = np.zeros((classes, classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)):
        cm[t, p] += 1
    return cm


def class_scores(cm: np.ndarray) -> dict[str, float]:
    """Accuracy, precision, recall, F1 from a confusion matrix (rows = truth).  Macro = plain mean over classes,
    weighted = mean weighted by class support (so weighted recall == accuracy).  The paper does not say which
    average Tables V / Figs. 11-12 use."""
    tp = np.diag(cm).astype(float)
    predicted, actual = cm.sum(axis=0).astype(float), cm.sum(axis=1).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    w = actual / actual.sum()
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "precision_macro": float(precision.mean()), "recall_macro": float(recall.mean()), "f1_macro": float(f1.mean()),
        "precision_weighted": float((w * precision).sum()), "recall_weighted": float((w * recall).sum()),
        "f1_weighted": float((w * f1).sum()),
    }


def roc_auc_ovr(y_true: np.ndarray, probabilities: np.ndarray, positive: int) -> float | None:
    """One-vs-rest AUC through the rank-sum (Mann-Whitney) identity with average ranks for ties:
    AUC = (sum of positive ranks - n_pos (n_pos + 1) / 2) / (n_pos n_neg)."""
    labels = np.asarray(y_true).reshape(-1) == positive
    scores = np.asarray(probabilities).reshape(-1, probabilities.shape[-1])[:, positive]
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores, method="average")
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cumulative_forecast(pred: np.ndarray, true: np.ndarray, horizon: int) -> dict[str, float]:
    """MSE and MAE over leads 1..h, all channels, in the units of the arrays (z-scored for the paper's tables)."""
    err = pred[:, :horizon] - true[:, :horizon]
    return {"mse": float(np.mean(err ** 2)), "mae": float(np.mean(np.abs(err)))}


def cumulative_classification(probabilities: np.ndarray, labels: np.ndarray, horizon: int) -> dict[str, float]:
    pred = probabilities[:, :horizon].argmax(axis=-1)
    return class_scores(confusion(labels[:, :horizon], pred))


# Paper, Tables IV and V, OmniFuser on Mudestreda (the Nonastreda data of this repo); MSE / MAE are z-scored.
PAPER_HORIZONS = (12, 24, 36, 48, 60, 72, 84, 96)
PAPER_OMNIFUSER = {
    "mse": (0.467, 0.514, 0.586, 0.686, 0.772, 0.830, 0.917, 0.902),
    "mae": (0.475, 0.509, 0.548, 0.590, 0.630, 0.654, 0.686, 0.693),
    "accuracy": (0.886, 0.928, 0.838, 0.866, 0.820, 0.863, 0.869, 0.863),
    "precision": (0.890, 0.922, 0.905, 0.913, 0.802, 0.862, 0.853, 0.842),
}
