"""NumPy reference for the front end of OmniFuser (arXiv 2511.01320v1, Sec. V-A and V-B) and for the whole forward pass.

Written from the paper's text, independent of ``omnifuser/``.  Extends ``c2f_reference.py`` (Blocks 6-8) and
``blocks9_11_reference.py`` (Blocks 9-11).  One sample, float64, sequences are (T, d).  [CHOICE] marks what the paper
leaves open and follows the repo, so that a parity test is meaningful; the register of choices is in
BLOCKS_12_NOTES.md.

    Eq. 1       positional_encoding(), sensor_embedding()           value MLP + sinusoidal position + period tables
    Eq. 2       depthwise_strided()                                 depthwise conv, kernel = stride, then W^(l)
    Eq. 3       rtd_block()                                         low-rank trend / seasonal split, feed-forward, restore T
                mrte()                                              Z^r = sum over resolutions of the restored sequences
    Eq. 4       compact_encoder()                                   (stand-in for EffNet: the repo's small CNN)
                gat_layer(), image_branch()                         GAT over the T frames, then position and period
    whole       full_forward()                                      front end -> Eq. 20 rounds -> Eq. 21-23 heads
    counting    expected_parameter_count()                          parameters implied by the dimensions alone
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

import blocks9_11_reference as b9
import c2f_reference as c2f

LEAKY_SLOPE = 0.2   # [CHOICE] GAT convention (Velickovic et al.); the paper says only "edge weights adaptively learned"


# =================================================================================================== Eq. 1
def positional_encoding(length: int, d: int) -> np.ndarray:
    """PE[t, 2i] = sin(t / 10000^(2i/d)), PE[t, 2i+1] = cos(t / 10000^(2i/d)): "sine and cosine functions with
    predefined frequencies" (the paper gives no formula; this is the standard one)."""
    position = np.arange(length, dtype=np.float64)[:, None]
    frequency = np.exp(np.arange(0, d, 2, dtype=np.float64) * (-math.log(10000.0) / d))
    angle = position * frequency
    pe = np.zeros((length, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle[:, : pe[:, 1::2].shape[1]])
    return pe


def _lookup(table: np.ndarray, values: np.ndarray | None, steps: int) -> np.ndarray:
    """Learnable embedding table lookup; a missing attribute uses row 0, an out-of-range code is clipped to the last row
    [CHOICE of the repo]."""
    if values is None:
        return np.repeat(table[:1], steps, axis=0)
    codes = np.clip(np.asarray(values, dtype=np.int64), 0, table.shape[0] - 1)
    if codes.ndim == 0:
        codes = np.full(steps, int(codes))
    return table[codes]


def sensor_embedding(force: np.ndarray, p: dict[str, Any], pe: np.ndarray, cycle=None, batch=None, relative=None):
    """Eq. 1:  H^r = MLP(x) + PE + (TE_c(c) + TE_b(b) + TE_p(p)).  MLP = two linear layers with a GELU between
    [CHOICE: the paper says "two stacked linear layers", the activation is not named]."""
    steps = force.shape[0]
    value = c2f.gelu(force @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]
    period = _lookup(p["cycle"], cycle, steps) + _lookup(p["batch"], batch, steps) + _lookup(p["relative"], relative, steps)
    return value + pe[:steps] + period


# =================================================================================================== Eq. 2-3
def depthwise_strided(x: np.ndarray, kernel: np.ndarray, stride: int) -> np.ndarray:
    """Eq. 2 before the channel projection:  out_t = sum_{tau<k} x_{t s + tau} * k_tau  (Hadamard product, k = s),
    t = 0 ... ceil(T/s) - 1, zero padding at the end when T is not a multiple of s.  kernel is (d, k)."""
    steps, d = x.shape
    out_steps = math.ceil(steps / stride)
    padded = np.zeros((out_steps * stride, d))
    padded[:steps] = x
    out = np.zeros((out_steps, d))
    for tau in range(stride):
        out += padded[tau::stride][:out_steps] * kernel[:, tau][None, :]
    return out


def _ffn(x: np.ndarray, q: dict[str, np.ndarray]) -> np.ndarray:
    return c2f.gelu(x @ q["w1"] + q["b1"]) @ q["w2"] + q["b2"]


def rtd_block(h: np.ndarray, q: dict[str, Any], stride: int, total_steps: int, decomp_mode: str = "branch"):
    """One Resolution-wise Temporal Decomposer (Fig. 5).  Returns (H_hat^(l), U_tilde^(l)).

    H_hat = depthwise_strided(H) W                       Eq. 2
    T = H_hat P Q,  S = H_hat - T                        Eq. 3
    U = T + S  (= H_hat; the sum cancels, so Eq. 3 as printed changes nothing)   decomp_mode "literal"
    [CHOICE] "branch" follows Fig. 5: one feed-forward network per branch, U = FFN_t(T) + FFN_s(S).
    U_tilde = FFD(U) in R^{T x d}: a linear map over time from T_l back to T, then GELU and a channel feed-forward
    [CHOICE: the paper says "an independent feed-forward module that restores the temporal length to T"].
    """
    down = depthwise_strided(h, q["dw"], stride) @ q["mix"]
    trend = (down @ q["p_left"]) @ q["p_right"]
    seasonal = down - trend
    if decomp_mode == "literal":
        decomposed = trend + seasonal
    elif decomp_mode == "branch":
        decomposed = _ffn(trend, q["trend"]) + _ffn(seasonal, q["seasonal"])
    else:
        raise ValueError(decomp_mode)
    up = q["time_up_w"] @ decomposed + q["time_up_b"][:, None]          # (T, T_l) @ (T_l, d)
    assert up.shape == (total_steps, h.shape[1])
    return down, _ffn(c2f.gelu(up), q["channel"])


def mrte(h: np.ndarray, blocks: list[dict[str, Any]], stride: int, decomp_mode: str = "branch") -> np.ndarray:
    """Z^r = sum_l U_tilde^(l).  [CHOICE] each block reads the downsampled output of the previous one (a pyramid with
    lengths T, ceil(T/s), ceil(T/s^2) ...); Eq. 2 writes the input of every block as H^r, which with s_l = 3 for all l
    would give L blocks of the same resolution."""
    total = np.zeros_like(h)
    current = h
    for q in blocks:
        current, restored = rtd_block(current, q, stride, h.shape[0], decomp_mode)
        total = total + restored
    return total


# =================================================================================================== Eq. 4
def conv2d(x: np.ndarray, w: np.ndarray, stride: int, pad: int, groups: int = 1) -> np.ndarray:
    """Cross-correlation like torch.nn.Conv2d without bias.  x (C, H, W), w (O, C/groups, k, k)."""
    channels, height, width = x.shape
    out_channels, per_group, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    ho, wo = (height + 2 * pad - k) // stride + 1, (width + 2 * pad - k) // stride + 1
    out = np.zeros((out_channels, ho, wo))
    cout = out_channels // groups
    for g in range(groups):
        for o in range(cout):
            for c in range(per_group):
                for i in range(k):
                    for j in range(k):
                        patch = xp[g * per_group + c, i:i + stride * ho:stride, j:j + stride * wo:stride]
                        out[g * cout + o] += w[g * cout + o, c, i, j] * patch
    return out


def _bn(x: np.ndarray, bn: dict[str, np.ndarray], eps: float = 1e-5) -> np.ndarray:
    scale = bn["weight"] / np.sqrt(bn["var"] + eps)
    return (x - bn["mean"][:, None, None]) * scale[:, None, None] + bn["bias"][:, None, None]


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def compact_encoder(image: np.ndarray, q: dict[str, Any]) -> np.ndarray:
    """The repo's small CNN used instead of EfficientNet-B0 for fast tests (NOT in the paper):
    conv3x3/2 (24) - BN - SiLU - depthwise conv3x3/2 (48, groups 24) - BN - SiLU - conv1x1 (96) - BN - SiLU - global mean -
    Linear(96, d).  Evaluation-mode BatchNorm."""
    x = _silu(_bn(conv2d(image, q["w1"], 2, 1), q["bn1"]))
    x = _silu(_bn(conv2d(x, q["w2"], 2, 1, groups=24), q["bn2"]))
    x = _silu(_bn(conv2d(x, q["w3"], 1, 0), q["bn3"]))
    return x.mean(axis=(1, 2)) @ q["proj_w"] + q["proj_b"]


def gat_layer(x: np.ndarray, w: np.ndarray, a_src: np.ndarray, a_dst: np.ndarray, heads: int):
    """Graph attention over a fully connected temporal graph (Velickovic et al.).  x (T, d), w (d, d) as (in, out),
    a_src, a_dst (heads, d/heads).  e_ij = LeakyReLU(a_src . Wh_i + a_dst . Wh_j), alpha = softmax over j, h_i' = sum_j
    alpha_ij Wh_j, heads concatenated.  The paper states only "fully connected temporal graph, edge weights adaptively
    learned"; the form of the score is [CHOICE]."""
    steps, d = x.shape
    hd = d // heads
    values = (x @ w).reshape(steps, heads, hd)
    source = np.einsum("thk,hk->ht", values, a_src)
    target = np.einsum("thk,hk->ht", values, a_dst)
    logits = source[:, :, None] + target[:, None, :]                       # (h, i, j)
    logits = np.where(logits >= 0, logits, LEAKY_SLOPE * logits)
    attention = c2f.softmax(logits, axis=-1)
    out = np.einsum("hij,jhk->ihk", attention, values)
    return out.reshape(steps, d), attention


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1.0)


def image_branch(encoded: np.ndarray, indices: np.ndarray, gat: list[dict[str, Any]], p: dict[str, Any],
                 pe: np.ndarray, cycle=None, batch=None, relative=None, heads: int = 4) -> np.ndarray:
    """Z^i.  encoded (n_images, d) are the per-image features; ``indices`` (T,) says which image each time step shows
    (the paper replicates the image over the steps of a pass; indexing is the same thing without running the CNN T
    times).  Then L GAT layers, each x <- LayerNorm(x + ELU(GAT(x))) [CHOICE: residual, ELU and LayerNorm are not in
    the paper], and finally position and period embeddings are added (Sec. V-B)."""
    steps = len(indices)
    x = encoded[np.asarray(indices)]
    for layer in gat:
        out, _ = gat_layer(x, layer["w"], layer["a_src"], layer["a_dst"], heads)
        x = c2f.layer_norm(x + _elu(out), layer["ln_g"], layer["ln_b"])
    period = _lookup(p["cycle"], cycle, steps) + _lookup(p["batch"], batch, steps) + _lookup(p["relative"], relative, steps)
    return x + pe[:steps] + period


# =================================================================================================== whole forward pass
def full_forward(w: dict[str, Any], force: np.ndarray, images: np.ndarray, indices: np.ndarray, cycle=None, batch=None,
                 relative=None, *, stride: int = 3, decomp_mode: str = "branch", heads: int = 4, head: str = "direct",
                 horizon: int, channels: int = 3, classes: int = 3, zero: str | None = None, **layer_options):
    """One sample from raw inputs to the heads: Eq. 1-4, Eq. 5-19 (``c2f.c2f_layer``), Eq. 20, Eq. 21-22.

    w holds the weights as built by ``tests/test_block12_fullmodel.extract_all`` (reference layout); ``w["pe"]``, when
    present, replaces the closed-form positional encoding (the repo stores it in float32)."""
    steps = force.shape[0]
    pe = w["pe"] if "pe" in w else positional_encoding(max(steps, 1), w["embedding"]["w2"].shape[1])
    h = sensor_embedding(force, w["embedding"], pe, cycle, batch, relative)
    z_r = mrte(h, w["rtd"], stride, decomp_mode)
    encoded = np.stack([compact_encoder(image, w["encoder"]) for image in images])
    z_i = image_branch(encoded, indices, w["gat"], w["image"], pe, cycle, batch, relative, heads)
    if zero == "sensor":
        z_r = np.zeros_like(z_r)
    elif zero == "image":
        z_i = np.zeros_like(z_i)
    fused = b9.refine(z_r, z_i, w["fusion"], w["refine_r"], w["refine_i"], heads=heads, **layer_options)
    z_t = fused[-1][-1]
    if head == "paper_literal":
        y, o = b9.head_paper_literal(z_t, *w["head_literal"], horizon)
    elif head == "direct":
        y, o = b9.head_direct(z_t, *w["head_direct"], horizon, channels, classes)
    else:
        y, o = b9.head_lead_conditioned(z_t, *w["head_lead"])
    return y, o


# =================================================================================================== counting
def expected_parameter_count(*, d: int = 32, input_len: int = 96, horizon: int = 24, channels: int = 3, classes: int = 3,
                             rtd_layers: int = 4, stride: int = 3, rank: int = 8, gat_layers: int = 4, heads: int = 4,
                             rounds: int = 2, landmarks: int = 4, fusion_rank: int = 8, share_c2f: bool = False,
                             pre_attn: bool = False, private_attn: str = "residual_ln", head: str = "lead_conditioned",
                             encoder: str = "compact") -> dict[str, int]:
    """Number of trainable parameters implied by the dimensions alone, per part.  Independent of ``models.py``: if the
    repo contains an extra or missing layer, the totals differ."""
    mha = 4 * d * d + 4 * d                                                    # in_proj (3d x d + 3d) + out_proj (d x d + d)
    embedding = channels * d + d + d * d + d + (128 + 8 + 16) * d
    ffn_branch = (d * 2 * d + 2 * d) + (2 * d * d + d)
    steps_in, rtd = input_len, 0
    for _ in range(rtd_layers):
        steps_out = math.ceil(steps_in / stride)
        rtd += d * stride + d * d + 2 * d * rank + 2 * ffn_branch + (steps_out * input_len + input_len) + 2 * (d * d + d)
        steps_in = steps_out
    if encoder == "compact":
        backbone = 3 * 24 * 9 + 2 * 24 + 24 * 9 * 2 + 2 * 48 + 48 * 96 + 2 * 96 + (96 * d + d)
    else:                                                                       # torchvision EfficientNet-B0 features + Linear
        backbone = 4_007_548 + (1280 * d + d)
    image = backbone + gat_layers * (d * d + 2 * d + 2 * d) + (128 + 8 + 16) * d
    layer = (2 * d * d + 2 * d * landmarks + 2 * d * d + 2 * mha + (4 * d if private_attn == "residual_ln" else 0)
             + 2 * (2 * d * d + d) + 2 * (2 * d * 3 + 2 * d) + 2 * (2 * d * d + d) + 2
             + (2 * d * fusion_rank + fusion_rank + fusion_rank * d + d))
    if pre_attn:
        layer += 2 * mha + 4 * d
    fusion = layer * (1 if share_c2f else rounds)
    refine = 2 * (rounds - 1) * d * d
    if head == "direct":
        heads_count = d * horizon * channels + horizon * channels + d * horizon * classes + horizon * classes
    else:
        heads_count = d * channels + channels + d * classes + classes
        if head == "lead_conditioned":
            heads_count += horizon * d + 2 * d * d + d
    parts = {"embedding": embedding, "mrte": rtd, "image_branch": image, "fusion": fusion, "refine": refine, "heads": heads_count}
    parts["total"] = sum(parts.values())
    return parts
