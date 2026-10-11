"""Block 12: the front end (Eq. 1-4: embedding, MRTE / RTD, image branch) and the whole model, checked end to end against
the NumPy reference in tests/frontend_reference.py, which is written from the paper and does not import ``omnifuser``.

    python tests/test_block12_fullmodel.py          # or: python -m pytest tests/test_block12_fullmodel.py

Which tests need what
* test_b12_ref_*            numpy / scipy only: the reference checks itself (hand values, explicit loops, algebra).
* the other tests build the real modules of ``omnifuser/models.py`` in float64 / eval mode and compare every stage with
  the reference to about 1e-8: embedding (Eq. 1), one RTD block and the whole MRTE (Eq. 2-3), the compact CNN, the graph
  attention layer, the image branch (Sec. V-B) and the complete model from raw inputs to the heads (Eq. 1-23).
* tests marked (autograd) need real PyTorch autograd and are skipped on the NumPy stand-in; tests marked (torchvision)
  need torchvision.
* test_b12_dataset_to_model_contract feeds real ``WindowDataset`` / ``collate_windows`` batches (synthetic Nonastreda
  folder) into the model and into the reference.
"""
from __future__ import annotations

import contextlib
import io
import itertools
import sys
import traceback
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts"))

import blocks9_11_reference as b9  # noqa: E402
import c2f_reference as c2f  # noqa: E402
import frontend_reference as fr  # noqa: E402
import test_c2f_blocks678 as t6  # noqa: E402
from test_alignment_patch import Workspace, make_nonastreda  # noqa: E402

try:
    import torch

    HAVE_TORCH = hasattr(torch, "nn")
    HAVE_AUTOGRAD = HAVE_TORCH and hasattr(torch, "autograd")
except ImportError:  # pragma: no cover
    HAVE_TORCH = HAVE_AUTOGRAD = False

try:
    import torchvision  # noqa: F401

    HAVE_TORCHVISION = HAVE_AUTOGRAD
except ImportError:
    HAVE_TORCHVISION = False

D, T, HEADS, CHANNELS, CLASSES, HORIZON = 32, 24, 4, 3, 3, 6
IMG_H, IMG_W = 16, 48            # small photos keep the NumPy reference CNN fast; the model accepts any size
HEAD_KINDS = ("paper_literal", "lead_conditioned", "direct")
_a = t6._a


def need_torch() -> None:
    if not HAVE_TORCH:
        raise unittest.SkipTest("PyTorch is not installed")


def need_autograd() -> None:
    need_torch()
    if not HAVE_AUTOGRAD:
        raise unittest.SkipTest("needs autograd")


# ====================================================================================================== builders
def make_config(**overrides):
    from omnifuser.config import ModelConfig

    settings = dict(d_model=D, input_len=T, horizon=HORIZON, force_channels=CHANNELS, num_classes=CLASSES,
                    num_gat_heads=HEADS, dropout=0.0, image_encoder="compact")
    settings.update(overrides)
    return ModelConfig(**settings)


def randomize_full(module, seed: int) -> None:
    """Reproducible random weights with activations of order one, for any module of ``omnifuser/models.py`` (BatchNorm
    scales, shifts and running statistics included, so that eval-mode BatchNorm is not the identity)."""
    import torch.nn as nn

    rng = np.random.default_rng(seed)
    batch_norms = [m for m in module.modules() if isinstance(m, nn.BatchNorm2d)]
    special = {id(p) for m in batch_norms for p in (m.weight, m.bias)}
    with torch.no_grad():
        for name, p in module.named_parameters():
            shape = tuple(p.shape)
            if id(p) in special:
                continue
            if "gate_mix" in name:
                value = rng.normal(0.0, 1.0, size=shape)
            elif "norm" in name and name.endswith("weight"):
                value = 1.0 + 0.2 * rng.standard_normal(shape)
            elif name.endswith("bias") or "norm" in name:
                value = 0.1 * rng.standard_normal(shape)
            else:
                fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else 1
                value = rng.standard_normal(shape) / np.sqrt(max(fan_in, 1))
            p.copy_(torch.as_tensor(np.asarray(value), dtype=p.dtype))
        for m in batch_norms:
            n = m.weight.shape[0]
            m.weight.copy_(torch.as_tensor(1.0 + 0.2 * rng.standard_normal(n), dtype=m.weight.dtype))
            m.bias.copy_(torch.as_tensor(0.1 * rng.standard_normal(n), dtype=m.bias.dtype))
            m.running_mean.copy_(torch.as_tensor(0.2 * rng.standard_normal(n), dtype=m.running_mean.dtype))
            m.running_var.copy_(torch.as_tensor(0.5 + rng.random(n), dtype=m.running_var.dtype))


def build_full(seed: int = 1, **overrides):
    import omnifuser.models as M

    model = M.OmniFuser(make_config(**overrides)).double().eval()
    randomize_full(model, seed)
    return model


def build_part(kind: str, seed: int = 1, **overrides):
    """A single front-end module in float64 / eval mode with random weights."""
    import omnifuser.models as M

    cfg = make_config(**overrides)
    module = {
        "embedding": lambda: M.TemporalEmbedding(cfg),
        "mrte": lambda: M.MultiResolutionTemporalExtractor(cfg),
        "gat": lambda: M.GraphAttentionLayer(cfg.d_model, cfg.num_gat_heads, 0.0),
        "encoder": lambda: M.ImageFeatureEncoder("compact", cfg.d_model),
        "image": lambda: M.ImageBranch(cfg),
    }[kind]().double().eval()
    randomize_full(module, seed)
    return module


# ================================================================================================== extraction
def _ffn(seq) -> dict:
    return {"w1": _a(seq[0].weight).T, "b1": _a(seq[0].bias), "w2": _a(seq[2].weight).T, "b2": _a(seq[2].bias)}


def _bn(m) -> dict:
    return {"weight": _a(m.weight), "bias": _a(m.bias), "mean": _a(m.running_mean), "var": _a(m.running_var)}


def extract_embedding(te) -> dict:
    return {"w1": _a(te.value[0].weight).T, "b1": _a(te.value[0].bias), "w2": _a(te.value[2].weight).T,
            "b2": _a(te.value[2].bias), "cycle": _a(te.cycle.weight), "batch": _a(te.batch.weight),
            "relative": _a(te.relative.weight)}


def extract_rtd_block(block) -> dict:
    return {"dw": _a(block.depthwise.weight)[:, 0, :], "mix": _a(block.mix.weight).T,
            "p_left": _a(block.low_rank_left), "p_right": _a(block.low_rank_right),
            "trend": _ffn(block.trend_ffn), "seasonal": _ffn(block.seasonal_ffn),
            "time_up_w": _a(block.time_up.weight), "time_up_b": _a(block.time_up.bias), "channel": _ffn(block.channel_ffn)}


def extract_encoder(enc) -> dict:
    f = enc.features
    return {"w1": _a(f[0].weight), "bn1": _bn(f[1]), "w2": _a(f[3].weight), "bn2": _bn(f[4]),
            "w3": _a(f[6].weight), "bn3": _bn(f[7]), "proj_w": _a(enc.projection.weight).T, "proj_b": _a(enc.projection.bias)}


def extract_gat(branch) -> list[dict]:
    return [{"w": _a(layer.projection.weight).T, "a_src": _a(layer.source_score), "a_dst": _a(layer.target_score),
             "ln_g": _a(norm.weight), "ln_b": _a(norm.bias)} for layer, norm in zip(branch.layers, branch.norms)]


def extract_all(model) -> dict:
    """All weights of ``OmniFuser`` in the layout of ``frontend_reference.full_forward`` ((in, out) matrices)."""
    cfg = model.config
    w = {
        "embedding": extract_embedding(model.temporal_embedding),
        "pe": _a(model.temporal_embedding.position_encoding),
        "rtd": [extract_rtd_block(b) for b in model.temporal_encoder.layers],
        "encoder": extract_encoder(model.image_branch.encoder),
        "gat": extract_gat(model.image_branch),
        "image": {k: _a(getattr(model.image_branch, k).weight) for k in ("cycle", "batch", "relative")},
        "fusion": [t6.extract_params(layer) for layer in model.fusion_layers],
        "refine_r": [_a(m.weight).T for m in model.refine_sensor],
        "refine_i": [_a(m.weight).T for m in model.refine_image],
    }
    if cfg.prediction_head == "paper_literal":
        w["head_literal"] = (_a(model.force_head.weight).T, _a(model.force_head.bias),
                             _a(model.classification_head.weight).T, _a(model.classification_head.bias))
    elif cfg.prediction_head == "direct":
        w["head_direct"] = (_a(model.direct_force.weight).T, _a(model.direct_force.bias),
                            _a(model.direct_class.weight).T, _a(model.direct_class.bias))
    else:
        w["head_lead"] = (_a(model.forecast_step_embedding.weight), _a(model.forecast_projection[0].weight).T,
                          _a(model.forecast_projection[0].bias), _a(model.force_head.weight).T,
                          _a(model.force_head.bias), _a(model.classification_head.weight).T,
                          _a(model.classification_head.bias))
    return w


def reference_options(model) -> dict:
    cfg = model.config
    return dict(stride=cfg.rtd_stride, decomp_mode=cfg.decomp_mode, heads=cfg.num_gat_heads, head=cfg.prediction_head,
                horizon=cfg.horizon, channels=cfg.force_channels, classes=cfg.num_classes, als_mode=cfg.als_mode,
                private_attn=cfg.private_attn, pre_attn=cfg.pre_attn)


# ====================================================================================================== inputs
def make_inputs(seed: int, batch: int = 3, steps: int = T, n_images: int = 3, codes: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    inputs = {
        "force": rng.standard_normal((batch, steps, CHANNELS)),
        "images": rng.standard_normal((batch, n_images, 3, IMG_H, IMG_W)),
        # a pass keeps showing one photo, so the index is non-decreasing along the window (as in collate_windows)
        "indices": np.sort(rng.integers(0, n_images, (batch, steps)), axis=1),
        "cycle": rng.integers(0, 128, (batch, steps)) if codes else None,
        "batch": rng.integers(0, 8, (batch, steps)) if codes else None,
        "relative": rng.integers(0, 16, (batch, steps)) if codes else None,
    }
    return inputs


def as_long(array):
    return None if array is None else torch.as_tensor(np.ascontiguousarray(array), dtype=torch.long)


def run_model(model, inputs: dict, **kwargs):
    with torch.no_grad():
        return model(t6.as_tensor(inputs["force"]), t6.as_tensor(inputs["images"]), as_long(inputs["indices"]),
                     as_long(inputs["cycle"]), as_long(inputs["batch"]), as_long(inputs["relative"]), **kwargs)


def reference_full(model, inputs: dict, zero: str | None = None):
    w, options = extract_all(model), reference_options(model)
    forces, logits = [], []
    for b in range(inputs["force"].shape[0]):
        pick = lambda key: None if inputs[key] is None else inputs[key][b]      # noqa: E731
        y, o = fr.full_forward(w, inputs["force"][b], inputs["images"][b], inputs["indices"][b], pick("cycle"),
                               pick("batch"), pick("relative"), zero=zero, **options)
        forces.append(y)
        logits.append(o)
    return np.stack(forces), np.stack(logits)


def assert_close(actual, expected, what: str, rtol: float = 1e-6, atol: float = 1e-8) -> float:
    actual = _a(actual) if hasattr(actual, "detach") else np.asarray(actual)
    assert actual.shape == expected.shape, (what, actual.shape, expected.shape)
    dif = float(np.abs(actual - expected).max())
    assert np.allclose(actual, expected, rtol=rtol, atol=atol), (what, dif)
    return dif


def count(*modules) -> int:
    seen, total = set(), 0
    for module in modules:
        for p in module.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += int(np.prod(tuple(p.shape)))
    return total


def actual_parts(model) -> dict:
    head_modules = [m for m in (model.forecast_step_embedding, model.forecast_projection, getattr(model, "force_head", None),
                                getattr(model, "classification_head", None), getattr(model, "direct_force", None),
                                getattr(model, "direct_class", None)) if m is not None]
    return {"embedding": count(model.temporal_embedding), "mrte": count(model.temporal_encoder),
            "image_branch": count(model.image_branch), "fusion": count(*model.fusion_layers),
            "refine": count(*model.refine_sensor, *model.refine_image), "heads": count(*head_modules)}


def implied_parts(cfg) -> dict:
    return fr.expected_parameter_count(
        d=cfg.d_model, input_len=cfg.input_len, horizon=cfg.horizon, channels=cfg.force_channels, classes=cfg.num_classes,
        rtd_layers=cfg.num_rtd_layers, stride=cfg.rtd_stride, rank=cfg.rtd_rank, gat_layers=cfg.num_gat_layers,
        heads=cfg.num_gat_heads, rounds=cfg.num_fusion_layers, landmarks=cfg.num_landmarks, fusion_rank=cfg.fusion_rank,
        share_c2f=cfg.share_c2f, pre_attn=cfg.pre_attn, private_attn=cfg.private_attn, head=cfg.prediction_head,
        encoder="compact" if cfg.image_encoder == "compact" else "efficientnet_b0")


# ===================================================================================== the reference checks itself
def test_b12_ref_positional_encoding_hand_values():
    pe = fr.positional_encoding(5, 4)
    assert np.allclose(pe[0], [0.0, 1.0, 0.0, 1.0])
    assert np.allclose(pe[1], [np.sin(1.0), np.cos(1.0), np.sin(0.01), np.cos(0.01)])      # 10000^(-2/4) = 0.01
    assert np.allclose(pe[4], [np.sin(4.0), np.cos(4.0), np.sin(0.04), np.cos(0.04)])
    big = fr.positional_encoding(96, 32)
    assert np.allclose((big[:, 0::2] ** 2 + big[:, 1::2] ** 2), 1.0)                       # sin^2 + cos^2 per frequency
    assert len({tuple(np.round(row, 9)) for row in big}) == 96                              # every position is different


def test_b12_ref_depthwise_strided_equals_the_explicit_sum():
    rng = np.random.default_rng(1)
    for steps, stride in ((12, 3), (10, 3), (7, 4), (5, 2), (1, 3)):
        x, kernel = rng.standard_normal((steps, 5)), rng.standard_normal((5, stride))
        got = fr.depthwise_strided(x, kernel, stride)
        expected = np.zeros(((steps + stride - 1) // stride, 5))
        for t in range(expected.shape[0]):
            for tau in range(stride):
                if t * stride + tau < steps:                      # zeros past the end of the window
                    expected[t] += x[t * stride + tau] * kernel[:, tau]
        assert got.shape == expected.shape and np.allclose(got, expected, atol=1e-14), (steps, stride)


def test_b12_ref_eq3_cancels_when_applied_literally_and_the_branches_do_not():
    rng = np.random.default_rng(2)
    d, rank, steps = 8, 2, 12
    q = {"dw": rng.standard_normal((d, 3)), "mix": rng.standard_normal((d, d)) / 3,
         "p_left": rng.standard_normal((d, rank)), "p_right": rng.standard_normal((rank, d))}
    for key in ("trend", "seasonal", "channel"):
        q[key] = {"w1": rng.standard_normal((d, 2 * d)) / 3, "b1": rng.standard_normal(2 * d) * 0.1,
                  "w2": rng.standard_normal((2 * d, d)) / 4, "b2": rng.standard_normal(d) * 0.1}
    q["channel"] = {"w1": rng.standard_normal((d, d)) / 3, "b1": np.zeros(d), "w2": rng.standard_normal((d, d)) / 3, "b2": np.zeros(d)}
    q["time_up_w"], q["time_up_b"] = rng.standard_normal((steps, 4)), rng.standard_normal(steps)
    h = rng.standard_normal((steps, d))
    down, literal = fr.rtd_block(h, q, 3, steps, "literal")
    # literal: U = T + S = H_hat, so the low-rank factors cannot matter
    other = dict(q, p_left=rng.standard_normal((d, rank)), p_right=rng.standard_normal((rank, d)))
    assert np.allclose(literal, fr.rtd_block(h, other, 3, steps, "literal")[1], atol=1e-12)
    _, branch = fr.rtd_block(h, q, 3, steps, "branch")
    _, branch_other = fr.rtd_block(h, other, 3, steps, "branch")
    assert not np.allclose(branch, branch_other, atol=1e-6) and not np.allclose(branch, literal, atol=1e-6)
    assert down.shape == (4, d)


def test_b12_ref_gat_is_a_row_stochastic_permutation_equivariant_map():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((9, D))
    w, a_src, a_dst = rng.standard_normal((D, D)) / np.sqrt(D), rng.standard_normal((HEADS, D // HEADS)), rng.standard_normal((HEADS, D // HEADS))
    out, attention = fr.gat_layer(x, w, a_src, a_dst, HEADS)
    assert attention.shape == (HEADS, 9, 9) and np.allclose(attention.sum(axis=-1), 1.0) and (attention > 0).all()
    perm = rng.permutation(9)
    out_perm, _ = fr.gat_layer(x[perm], w, a_src, a_dst, HEADS)       # fully connected graph, no positions inside the layer
    assert np.allclose(out_perm, out[perm], atol=1e-12)
    same = np.repeat(x[:1], 9, axis=0)                                 # identical frames -> uniform attention
    _, uniform = fr.gat_layer(same, w, a_src, a_dst, HEADS)
    assert np.allclose(uniform, 1.0 / 9)


def test_b12_ref_image_branch_without_gat_layers_is_gather_plus_position_plus_period():
    rng = np.random.default_rng(4)
    encoded, indices = rng.standard_normal((3, D)), np.array([0, 0, 1, 1, 1, 2])
    tables = {k: rng.standard_normal((n, D)) for k, n in (("cycle", 128), ("batch", 8), ("relative", 16))}
    pe = fr.positional_encoding(6, D)
    got = fr.image_branch(encoded, indices, [], tables, pe, cycle=np.arange(6), batch=np.ones(6, int), relative=np.zeros(6, int))
    expected = encoded[indices] + pe + tables["cycle"][np.arange(6)] + tables["batch"][1] + tables["relative"][0]
    assert np.allclose(got, expected, atol=1e-14)


def test_b12_ref_parameter_count_at_the_paper_settings():
    parts = fr.expected_parameter_count()                                # d 32, T 96, P 24, compact encoder, lead-conditioned
    assert parts == {"embedding": 6_048, "mrte": 53_600, "image_branch": 18_600, "fusion": 45_140, "refine": 2_048,
                     "heads": 3_046, "total": 128_482}
    paper = fr.expected_parameter_count(encoder="efficientnet_b0", head="direct", horizon=96)
    assert paper["image_branch"] == 4_007_548 + 1280 * 32 + 32 + 4 * (32 * 32 + 2 * 32 + 2 * 32) + 152 * 32
    assert paper["heads"] == 32 * 96 * 3 + 96 * 3 + 32 * 96 * 3 + 96 * 3
    assert fr.expected_parameter_count(share_c2f=True)["fusion"] == 22_570 and fr.expected_parameter_count(rounds=3)["refine"] == 4_096


# ================================================================================ Eq. 1: the embedding (real module)
def test_b12_positional_encoding_buffer_is_the_closed_form():
    need_torch()
    from omnifuser.models import sinusoidal_position_encoding

    got = _a(sinusoidal_position_encoding(96, 32))
    assert np.allclose(got, fr.positional_encoding(96, 32), atol=2e-5)       # float32 rounding of the angle t * f (<= 3e-6 here)
    assert np.allclose(got[0, 0::2], 0) and np.allclose(got[0, 1::2], 1)
    odd = _a(sinusoidal_position_encoding(10, 7))                             # odd widths do not crash
    assert odd.shape == (10, 7)
    te = build_part("embedding")
    assert "position_encoding" not in te.state_dict()                          # non-persistent: not part of checkpoints


def test_b12_embedding_matches_eq1_with_every_kind_of_period_code():
    need_torch()
    te = build_part("embedding", seed=2)
    w, pe = extract_embedding(te), _a(te.position_encoding)
    inputs = make_inputs(3, batch=3)
    per_sample = np.array([5, 77, 300])                                         # one code per sample (expanded over time)
    cases = {
        "all codes (B, T)": (inputs["cycle"], inputs["batch"], inputs["relative"]),
        "no codes": (None, None, None),
        "codes (B,)": (per_sample, np.array([1, 2, 3]), np.array([0, 15, 40])),
        "out of range": (np.full((3, T), 999), np.full((3, T), -4), np.full((3, T), 16)),
    }
    for label, (c, b, r) in cases.items():
        with torch.no_grad():
            out = te(t6.as_tensor(inputs["force"]), as_long(c), as_long(b), as_long(r))
        assert tuple(out.shape) == (3, T, D)
        for s in range(3):
            def pick(v):
                if v is None:
                    return None
                v = np.asarray(v)
                return v[s] if v.ndim == 2 else int(v[s])
            expected = fr.sensor_embedding(inputs["force"][s], w, pe, pick(c), pick(b), pick(r))
            assert_close(out[s], expected, f"embedding {label} sample {s}")
    with torch.no_grad():
        try:
            te(t6.as_tensor(np.zeros((1, T + 1, CHANNELS))))
        except ValueError:
            pass
        else:
            raise AssertionError("a window longer than input_len must be rejected")


def test_b12_embedding_depends_on_the_period_codes_and_the_position():
    need_torch()
    te = build_part("embedding", seed=5)
    force = t6.as_tensor(np.zeros((1, T, CHANNELS)))
    with torch.no_grad():
        base = _a(te(force))
        shifted = _a(te(force, as_long(np.full((1, T), 3))))
    assert not np.allclose(base, shifted)
    assert not np.allclose(base[0, 0], base[0, 1])                              # constant force, different positions


# ==================================================================================== Eq. 2-3: RTD block and MRTE
def test_b12_rtd_block_matches_reference_for_every_window_length_and_decomposition():
    need_torch()
    worst = 0.0
    for steps, mode in itertools.product((96, 24, 25, 26, 7), ("branch", "literal")):
        mrte = build_part("mrte", seed=steps, input_len=steps, decomp_mode=mode, num_rtd_layers=2)
        h = np.random.default_rng(steps).standard_normal((2, steps, D))
        with torch.no_grad():
            down, up = mrte.layers[0](t6.as_tensor(h))
        q = extract_rtd_block(mrte.layers[0])
        for s in range(2):
            ref_down, ref_up = fr.rtd_block(h[s], q, 3, steps, mode)
            worst = max(worst, assert_close(down[s], ref_down, f"down T={steps} {mode}"),
                        assert_close(up[s], ref_up, f"up T={steps} {mode}"))
        assert tuple(down.shape) == (2, -(-steps // 3), D) and tuple(up.shape) == (2, steps, D)
    assert worst < 1e-8, worst


def test_b12_rtd_pyramid_has_the_documented_lengths():
    need_torch()
    mrte = build_part("mrte", input_len=96)
    assert [b.input_steps for b in mrte.layers] == [96, 32, 11, 4]
    assert [b.output_steps for b in mrte.layers] == [32, 11, 4, 2]
    assert [tuple(b.time_up.weight.shape) for b in mrte.layers] == [(96, 32), (96, 11), (96, 4), (96, 2)]
    block = mrte.layers[1]
    try:
        block(t6.as_tensor(np.zeros((1, 33, D))))
    except ValueError:
        pass
    else:
        raise AssertionError("an RTD block must reject a sequence of the wrong length")


def test_b12_mrte_matches_reference_for_several_strides_depths_and_lengths():
    need_torch()
    for steps, stride, layers, mode in ((96, 3, 4, "branch"), (50, 3, 3, "branch"), (24, 2, 3, "literal"),
                                        (30, 4, 2, "branch"), (10, 3, 4, "branch"), (96, 3, 4, "literal"),
                                        (24, 1, 2, "branch")):          # s = 1: the paper's sensitivity grid starts there
        mrte = build_part("mrte", seed=steps + stride, input_len=steps, rtd_stride=stride, num_rtd_layers=layers,
                          decomp_mode=mode)
        h = np.random.default_rng(steps).standard_normal((2, steps, D))
        with torch.no_grad():
            out = mrte(t6.as_tensor(h))
        blocks = [extract_rtd_block(b) for b in mrte.layers]
        for s in range(2):
            assert_close(out[s], fr.mrte(h[s], blocks, stride, mode), f"MRTE T={steps} s={stride} L={layers} {mode}")
        assert tuple(out.shape) == (2, steps, D)


def test_b12_literal_mode_makes_eq3_a_no_op_branch_mode_does_not():
    need_torch()
    h = np.random.default_rng(7).standard_normal((1, T, D))
    outputs = {}
    for mode in ("literal", "branch"):
        mrte = build_part("mrte", seed=8, decomp_mode=mode, num_rtd_layers=2)
        with torch.no_grad():
            before = _a(mrte(t6.as_tensor(h)))
            block = mrte.layers[0]
            block.low_rank_left.copy_(torch.as_tensor(np.random.default_rng(9).standard_normal(tuple(block.low_rank_left.shape))))
            block.low_rank_right.copy_(torch.as_tensor(np.random.default_rng(10).standard_normal(tuple(block.low_rank_right.shape))))
            after = _a(mrte(t6.as_tensor(h)))
        outputs[mode] = np.abs(before - after).max()
    assert outputs["literal"] < 1e-12, outputs            # T + S = H_hat whatever the rank-r split is
    assert outputs["branch"] > 1e-3, outputs


def test_b12_trend_has_rank_r_and_trend_plus_seasonal_is_the_downsampled_signal():
    need_torch()
    mrte = build_part("mrte", seed=11, num_rtd_layers=1, input_len=96)
    block, seen = mrte.layers[0], {}
    handles = [block.trend_ffn.register_forward_hook(lambda m, a, o: seen.__setitem__("trend", _a(a[0][0]))),
               block.seasonal_ffn.register_forward_hook(lambda m, a, o: seen.__setitem__("seasonal", _a(a[0][0])))]
    h = np.random.default_rng(12).standard_normal((1, 96, D))
    with torch.no_grad():
        down, _ = block(t6.as_tensor(h))
    for handle in handles:
        handle.remove()
    assert np.linalg.matrix_rank(seen["trend"], tol=1e-9) == 8 == block.low_rank_left.shape[1]
    assert np.linalg.matrix_rank(seen["seasonal"], tol=1e-9) > 8                   # the remainder is not low rank
    assert np.allclose(seen["trend"] + seen["seasonal"], _a(down[0]), atol=1e-12)


# =================================================================================== Eq. 4 and Sec. V-B: image side
def test_b12_compact_encoder_matches_reference():
    need_torch()
    enc = build_part("encoder", seed=13)
    images = np.random.default_rng(14).standard_normal((2, 3, 3, IMG_H, IMG_W))
    with torch.no_grad():
        out = enc(t6.as_tensor(images))
    q = extract_encoder(enc)
    assert tuple(out.shape) == (2, 3, D)
    for b in range(2):
        for n in range(3):
            assert_close(out[b, n], fr.compact_encoder(images[b, n], q), f"encoder image ({b},{n})")
    with torch.no_grad():
        empty = enc(t6.as_tensor(np.zeros((2, 0, 3, IMG_H, IMG_W))))
    assert tuple(empty.shape) == (2, 0, D)


def test_b12_compact_encoder_is_applied_to_each_photo_independently_in_eval_mode():
    need_torch()
    enc = build_part("encoder", seed=15)
    images = np.random.default_rng(16).standard_normal((2, 2, 3, IMG_H, IMG_W))
    with torch.no_grad():
        both = _a(enc(t6.as_tensor(images)))
        alone = _a(enc(t6.as_tensor(images[1:, 1:])))
    assert np.allclose(both[1, 1], alone[0, 0], rtol=1e-9, atol=1e-11)


def test_b12_graph_attention_layer_matches_reference():
    need_torch()
    gat = build_part("gat", seed=17)
    x = np.random.default_rng(18).standard_normal((3, T, D))
    with torch.no_grad():
        out, attention = gat(t6.as_tensor(x))
    assert tuple(out.shape) == (3, T, D) and tuple(attention.shape) == (3, HEADS, T, T)
    for b in range(3):
        ref_out, ref_attention = fr.gat_layer(x[b], _a(gat.projection.weight).T, _a(gat.source_score), _a(gat.target_score), HEADS)
        assert_close(out[b], ref_out, f"GAT output {b}")
        assert_close(attention[b], ref_attention, f"GAT attention {b}")
    assert np.allclose(_a(attention).sum(axis=-1), 1.0)


def test_b12_image_branch_matches_reference_with_repeated_and_arbitrary_indices():
    need_torch()
    worst = 0.0
    for seed, pass_like in ((19, True), (20, False)):
        branch = build_part("image", seed=seed)
        inputs = make_inputs(seed, batch=3, n_images=3)
        if not pass_like:
            inputs["indices"] = np.random.default_rng(seed).integers(0, 3, (3, T))
        with torch.no_grad():
            out = branch(t6.as_tensor(inputs["images"]), as_long(inputs["indices"]), as_long(inputs["cycle"]),
                         as_long(inputs["batch"]), as_long(inputs["relative"]))
        assert tuple(out.shape) == (3, T, D)
        gat, q = extract_gat(branch), extract_encoder(branch.encoder)
        tables = {k: _a(getattr(branch, k).weight) for k in ("cycle", "batch", "relative")}
        pe = _a(branch.position_encoding)
        for s in range(3):
            encoded = np.stack([fr.compact_encoder(image, q) for image in inputs["images"][s]])
            expected = fr.image_branch(encoded, inputs["indices"][s], gat, tables, pe, inputs["cycle"][s],
                                       inputs["batch"][s], inputs["relative"][s], HEADS)
            worst = max(worst, assert_close(out[s], expected, f"image branch sample {s} pass_like={pass_like}"))
    assert worst < 1e-8


def test_b12_image_branch_rejects_bad_indices_and_equals_the_papers_replication():
    need_torch()
    branch = build_part("image", seed=21)
    inputs = make_inputs(22, batch=2, steps=10, n_images=2)
    images = t6.as_tensor(inputs["images"])
    for bad in (np.full((2, 10), 2), np.full((2, 10), -1)):
        try:
            branch(images, as_long(bad))
        except ValueError:
            pass
        else:
            raise AssertionError("an image index outside the photos of the window must be rejected")
    # The paper repeats the photo over the steps of its pass and runs the CNN on each repeated copy.  Gathering the
    # per-photo features by index is the same computation (eval mode: BatchNorm uses running statistics).
    replicated = np.stack([inputs["images"][b][inputs["indices"][b]] for b in range(2)])            # (B, T, 3, H, W)
    with torch.no_grad():
        gathered = _a(branch(images, as_long(inputs["indices"])))
        copies = _a(branch(t6.as_tensor(replicated), as_long(np.tile(np.arange(10), (2, 1)))))
    assert np.allclose(gathered, copies, rtol=1e-9, atol=1e-11)


# ================================================================================================ the whole model
def test_b12_full_model_matches_reference_for_every_head_round_count_and_sharing():
    need_torch()
    worst = 0.0
    for head, rounds, share in itertools.product(HEAD_KINDS, (1, 2), (False, True)):
        if rounds == 1 and share:
            continue
        model = build_full(seed=30 + rounds, prediction_head=head, num_fusion_layers=rounds, share_c2f=share)
        inputs = make_inputs(31)
        out = run_model(model, inputs)
        force, logits = reference_full(model, inputs)
        assert tuple(out["force"].shape) == (3, HORIZON, CHANNELS) and tuple(out["class_logits"].shape) == (3, HORIZON, CLASSES)
        worst = max(worst, assert_close(out["force"], force, f"force {head} n={rounds} share={share}"),
                    assert_close(out["class_logits"], logits, f"logits {head} n={rounds} share={share}"))
    assert worst < 1e-8, worst


def test_b12_full_model_matches_reference_for_the_front_end_and_layer_options():
    need_torch()
    for mode, options in itertools.product(("branch", "literal"), (
            {}, {"als_mode": "k_sum", "private_attn": "plain", "pre_attn": True}, {"als_mode": "k_colnorm"})):
        model = build_full(seed=40, prediction_head="direct", decomp_mode=mode, **options)
        inputs = make_inputs(41)
        out = run_model(model, inputs)
        force, logits = reference_full(model, inputs)
        assert_close(out["force"], force, f"force {mode} {options}")
        assert_close(out["class_logits"], logits, f"logits {mode} {options}")


def test_b12_full_model_matches_reference_without_period_codes_and_with_a_single_photo():
    need_torch()
    model = build_full(seed=42, prediction_head="direct")
    for codes, n_images in ((False, 3), (True, 1)):
        inputs = make_inputs(43, n_images=n_images, codes=codes)
        out = run_model(model, inputs)
        force, logits = reference_full(model, inputs)
        assert_close(out["force"], force, f"force codes={codes} n={n_images}")
        assert_close(out["class_logits"], logits, f"logits codes={codes} n={n_images}")


def test_b12_full_model_drop_modality_equals_a_zero_stream_in_the_reference():
    need_torch()
    model = build_full(seed=44, prediction_head="direct")
    inputs = make_inputs(45)
    for which in ("sensor", "image"):
        out = run_model(model, inputs, drop_modality=which)
        force, logits = reference_full(model, inputs, zero=which)
        assert_close(out["force"], force, f"force without {which}")
        assert_close(out["class_logits"], logits, f"logits without {which}")
    both = run_model(model, inputs)
    assert not np.allclose(_a(both["force"]), _a(run_model(model, inputs, drop_modality="image")["force"]))


def test_b12_full_model_at_the_paper_dimensions_T96_d32_L4_s3_r8():
    need_torch()
    cfg = make_config(input_len=96, horizon=24, prediction_head="direct")
    assert (cfg.d_model, cfg.num_rtd_layers, cfg.rtd_stride, cfg.rtd_rank, cfg.num_gat_layers, cfg.num_gat_heads,
            cfg.num_fusion_layers, cfg.num_landmarks, cfg.fusion_rank) == (32, 4, 3, 8, 4, 4, 2, 4, 8)
    model = build_full(seed=46, input_len=96, horizon=24, prediction_head="direct")
    inputs = make_inputs(47, batch=2, steps=96, n_images=2)
    out = run_model(model, inputs)
    force, logits = reference_full(model, inputs)
    assert tuple(out["force"].shape) == (2, 24, 3) and tuple(out["class_logits"].shape) == (2, 24, 3)
    assert_close(out["force"], force, "force at T=96")
    assert_close(out["class_logits"], logits, "logits at T=96")
    assert np.isfinite(_a(out["force"])).all() and float(_a(out["alignment_loss"])) >= 0.0


def test_b12_outputs_do_not_depend_on_the_other_samples_of_the_batch():
    need_torch()
    model = build_full(seed=48, prediction_head="lead_conditioned")
    inputs = make_inputs(49, batch=3)
    whole = run_model(model, inputs)
    for b in range(3):
        single = {k: (None if v is None else v[b:b + 1]) for k, v in inputs.items()}
        alone = run_model(model, single)
        assert np.allclose(_a(alone["force"])[0], _a(whole["force"])[b], rtol=1e-9, atol=1e-11), b
        assert np.allclose(_a(alone["class_logits"])[0], _a(whole["class_logits"])[b], rtol=1e-9, atol=1e-11), b


def test_b12_inference_is_deterministic_and_the_forecast_reads_only_its_inputs():
    need_torch()
    model = build_full(seed=50, prediction_head="direct", dropout=0.3)           # dropout is inactive in eval mode
    inputs = make_inputs(51)
    first, second = run_model(model, inputs), run_model(model, inputs)
    assert np.array_equal(_a(first["force"]), _a(second["force"])) and np.array_equal(_a(first["class_logits"]), _a(second["class_logits"]))
    changed = dict(inputs, force=inputs["force"] + 0.5)
    assert not np.allclose(_a(run_model(model, changed)["force"]), _a(first["force"]))           # the force matters
    changed = dict(inputs, images=inputs["images"] * -1.0)
    assert not np.allclose(_a(run_model(model, changed)["force"]), _a(first["force"]))           # the photos matter
    unused = dict(inputs, images=np.concatenate([inputs["images"], np.random.default_rng(1).standard_normal(inputs["images"][:, :1].shape)], axis=1))
    assert np.allclose(_a(run_model(model, unused)["force"]), _a(first["force"]), rtol=1e-9, atol=1e-11)   # a photo nobody points at is ignored


def test_b12_train_mode_dropout_changes_the_output_eval_mode_does_not():
    need_autograd()
    model = build_full(seed=52, prediction_head="direct", dropout=0.5)
    inputs = make_inputs(53)
    torch.manual_seed(0)
    model.train()
    a, b = run_model(model, inputs), run_model(model, inputs)
    assert not np.allclose(_a(a["force"]), _a(b["force"]))
    model.eval()
    assert np.array_equal(_a(run_model(model, inputs)["force"]), _a(run_model(model, inputs)["force"]))


def test_b12_input_checks_reject_malformed_batches():
    need_torch()
    model = build_full(seed=54)
    inputs = make_inputs(55, batch=2)
    force, images, indices = t6.as_tensor(inputs["force"]), t6.as_tensor(inputs["images"]), as_long(inputs["indices"])
    bad = {
        "window of the wrong length": (t6.as_tensor(inputs["force"][:, :T - 1]), images, indices[:, :T - 1]),
        "force without a batch axis": (force[0], images, indices),
        "images of the wrong rank": (force, images[:, 0], indices),
        "images of another batch size": (force, images[:1], indices),
    }
    for label, args in bad.items():
        try:
            model(*args)
        except (ValueError, RuntimeError, IndexError, AssertionError):
            continue
        raise AssertionError(f"{label} must be rejected")


def test_b12_float32_forward_stays_close_to_the_float64_reference():
    need_torch()
    import omnifuser.models as M

    model64 = build_full(seed=56, prediction_head="direct")
    model32 = M.OmniFuser(make_config(prediction_head="direct")).eval()
    model32.load_state_dict(model64.state_dict())
    inputs = make_inputs(57)
    with torch.no_grad():
        out32 = model32(torch.as_tensor(inputs["force"], dtype=torch.float32), torch.as_tensor(inputs["images"], dtype=torch.float32),
                        as_long(inputs["indices"]), as_long(inputs["cycle"]), as_long(inputs["batch"]), as_long(inputs["relative"]))
    force, logits = reference_full(model64, inputs)
    scale = max(np.abs(force).max(), np.abs(logits).max(), 1.0)
    assert np.abs(_a(out32["force"]) - force).max() < 2e-4 * scale
    assert np.abs(_a(out32["class_logits"]) - logits).max() < 2e-4 * scale


# ================================================================================== parameters and checkpoints
def test_b12_parameter_counts_follow_from_the_dimensions():
    need_torch()
    cases = [
        {}, {"prediction_head": "direct"}, {"prediction_head": "paper_literal"}, {"num_fusion_layers": 1},
        {"num_fusion_layers": 3}, {"num_fusion_layers": 3, "share_c2f": True}, {"pre_attn": True},
        {"private_attn": "plain"}, {"input_len": 50}, {"input_len": 96, "horizon": 24, "prediction_head": "direct"},
        {"d_model": 16}, {"num_rtd_layers": 2, "rtd_stride": 2}, {"num_rtd_layers": 2, "rtd_stride": 1}, {"num_landmarks": 6, "fusion_rank": 4},
        {"num_gat_layers": 2, "rtd_rank": 4}, {"decomp_mode": "literal"},
    ]
    for overrides in cases:
        model = build_full(seed=60, **overrides)
        expected = implied_parts(model.config)
        actual = actual_parts(model)
        for part, number in actual.items():
            assert number == expected[part], (overrides, part, number, expected[part])
        assert count(model) == expected["total"], (overrides, count(model), expected["total"])


def test_b12_default_model_has_128482_parameters_with_the_compact_encoder():
    need_torch()
    import omnifuser.models as M
    from omnifuser.config import ModelConfig

    model = M.OmniFuser(ModelConfig(image_encoder="compact"))
    assert actual_parts(model) == {"embedding": 6_048, "mrte": 53_600, "image_branch": 18_600, "fusion": 45_140,
                                   "refine": 2_048, "heads": 3_046}
    assert count(model) == 128_482


def test_b12_efficientnet_b0_backbone_has_the_torchvision_parameter_count():
    """(torchvision) 'features' of torchvision's EfficientNet-B0 = 5,288,548 - 1,281,000 (the 1000-class classifier)."""
    if not HAVE_TORCHVISION:
        raise unittest.SkipTest("needs torchvision")
    import omnifuser.models as M

    enc = M.ImageFeatureEncoder("efficientnet_b0", 32)
    assert count(enc.features) == 4_007_548 and count(enc.projection) == 1280 * 32 + 32
    with torch.no_grad():
        out = enc.eval()(torch.zeros(1, 2, 3, 64, 96))
    assert tuple(out.shape) == (1, 2, 32)
    full = M.OmniFuser(make_config(image_encoder="efficientnet_b0", prediction_head="direct", horizon=24, input_len=96))
    assert count(full) == implied_parts(full.config)["total"]


def test_b12_state_dict_round_trip_reproduces_the_outputs():
    need_torch()
    import omnifuser.models as M

    model = build_full(seed=61, prediction_head="direct")
    state = model.state_dict()
    assert not any("position_encoding" in key for key in state)                  # rebuilt from the config, not stored
    assert any(key.endswith("running_mean") for key in state) and any(key.endswith("running_var") for key in state)
    clone = M.OmniFuser(make_config(prediction_head="direct")).double().eval()
    clone.load_state_dict(state)
    inputs = make_inputs(62)
    assert np.array_equal(_a(run_model(model, inputs)["force"]), _a(run_model(clone, inputs)["force"]))
    assert set(state) == set(clone.state_dict())
    if hasattr(torch, "save") and hasattr(torch, "load"):
        with Workspace() as tmp:
            torch.save(state, tmp / "checkpoint.pt")
            restored = M.OmniFuser(make_config(prediction_head="direct")).double().eval()
            restored.load_state_dict(torch.load(tmp / "checkpoint.pt"))
        assert np.array_equal(_a(run_model(model, inputs)["class_logits"]), _a(run_model(restored, inputs)["class_logits"]))


def test_b12_model_summary_script_agrees_with_the_dimensions():
    need_torch()
    import model_summary

    for argv in (["--image-encoder", "compact", "--horizon", "12"],
                 ["--image-encoder", "compact", "--horizon", "24", "--prediction-head", "direct", "--rounds", "3", "--share-c2f"]):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            result = model_summary.main(argv + ["--check", "--height", str(IMG_H), "--width", str(IMG_W)])
        assert result["total"] == result["expected"]["total"], argv
        stages = [row[0] for row in result["rows"]]
        assert stages[0] == "temporal embedding" and stages[1].startswith("MRTE") and any(s.startswith("heads") for s in stages)
        assert "(match)" in printed.getvalue() and sum(row[2] for row in result["rows"]) == result["total"]


def test_b12_architecture_flags_reach_the_model_config():
    need_torch()                                           # omnifuser.cli imports the training module
    from omnifuser.cli import ARCHITECTURE_OPTIONS, build_parser, protocol_kwargs
    from omnifuser.config import ModelConfig

    parser, base = build_parser(), ["train", "--data-dir", "x", "--output-dir", "o"]
    untouched = protocol_kwargs(parser.parse_args(base))["config_overrides"]
    assert not set(ARCHITECTURE_OPTIONS.values()) & set(untouched), untouched      # unset flags change nothing
    flags = ["--decomp-mode", "literal", "--rtd-layers", "3", "--rtd-stride", "2", "--rtd-rank", "4", "--gat-layers", "2",
             "--gat-heads", "2", "--fusion-layers", "3", "--landmarks", "6", "--fusion-rank", "16", "--dropout", "0.2"]
    overrides = protocol_kwargs(parser.parse_args(base + flags))["config_overrides"]
    assert {k: overrides[k] for k in ARCHITECTURE_OPTIONS.values()} == {
        "decomp_mode": "literal", "num_rtd_layers": 3, "rtd_stride": 2, "rtd_rank": 4, "num_gat_layers": 2,
        "num_gat_heads": 2, "num_fusion_layers": 3, "num_landmarks": 6, "fusion_rank": 16, "dropout": 0.2}
    config = ModelConfig(d_model=32, input_len=96, horizon=24, image_encoder="compact", prediction_head="direct", **overrides)
    assert (config.num_rtd_layers, config.rtd_stride, config.decomp_mode, config.num_fusion_layers) == (3, 2, "literal", 3)
    import omnifuser.models as M

    model = M.OmniFuser(config)
    assert len(model.temporal_encoder.layers) == 3 and model.temporal_encoder.layers[0].stride == 2
    assert [b.output_steps for b in model.temporal_encoder.layers] == [48, 24, 12]
    assert len(model.image_branch.layers) == 2 and model.image_branch.layers[0].heads == 2
    assert len(model.fusion_layers) == 3 and len(model.refine_sensor) == 2
    assert tuple(model.fusion_layers[0].score_r.weight.shape) == (6, 32)
    assert count(model) == implied_parts(config)["total"]


# ============================================================================ data -> model contract (real batches)
def test_b12_dataset_to_model_contract():
    need_torch()
    from omnifuser.data import (ForceNormalizer, WindowDataset, collate_windows, load_nonastreda, make_tool_timelines)

    with Workspace() as tmp:
        make_nonastreda(tmp, tools=(1, 2, 3), runs=(1, 2), length=500, seed=7)
        records = load_nonastreda(tmp, trim_pass=False)
        timelines = make_tool_timelines(records, image_lag=1)
        normalizer = ForceNormalizer.fit(timelines)
        data = WindowDataset(timelines, normalizer, input_len=T, horizon=HORIZON, stride=40)
        batch = collate_windows([data[i] for i in (0, 5, 11, len(data) - 1)], height=IMG_H, width=IMG_W)
    n = len(batch["force"])
    assert tuple(batch["force"].shape) == (n, T, 3) and batch["images"].shape[2:] == (3, IMG_H, IMG_W)
    assert tuple(batch["image_indices"].shape) == (n, T) and int(batch["image_indices"].max()) < batch["images"].shape[1]
    assert tuple(batch["force_target"].shape) == (n, HORIZON, 3) and tuple(batch["class_target"].shape) == (n, HORIZON)
    for key in ("cycle", "batch", "relative"):
        assert tuple(batch[key].shape) == (n, T)
    model = build_full(seed=70, prediction_head="direct")
    inputs = {"force": _a(batch["force"]), "images": _a(batch["images"]), "indices": batch["image_indices"].numpy(),
              "cycle": batch["cycle"].numpy(), "batch": batch["batch"].numpy(), "relative": batch["relative"].numpy()}
    out = run_model(model, inputs)
    force, logits = reference_full(model, inputs)
    assert tuple(out["force"].shape) == tuple(batch["force_target"].shape)
    assert_close(out["force"], force, "force on real batches")
    assert_close(out["class_logits"], logits, "logits on real batches")
    from omnifuser.models import multitask_loss

    total, parts = multitask_loss(out, t6.as_tensor(_a(batch["force_target"])), batch["class_target"])
    expected = b9.multitask_loss(force, _a(batch["force_target"]), logits, batch["class_target"].numpy())
    assert np.isclose(float(total), expected["loss"], rtol=1e-8)


# ================================================================================================== training (autograd)
def test_b12_gradients_reach_every_parameter_of_the_whole_model():
    need_autograd()
    for head in HEAD_KINDS:
        model = build_full(seed=80, prediction_head=head, dropout=0.1)
        model.train()
        inputs = make_inputs(81)
        torch.manual_seed(1)
        out = model(t6.as_tensor(inputs["force"]), t6.as_tensor(inputs["images"]), as_long(inputs["indices"]),
                    as_long(inputs["cycle"]), as_long(inputs["batch"]), as_long(inputs["relative"]))
        (out["force"].square().mean() + out["class_logits"].square().mean()).backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (head, name)
        for name in ("temporal_embedding.value.0.weight", "temporal_encoder.layers.0.depthwise.weight",
                     "temporal_encoder.layers.3.time_up.weight", "image_branch.layers.0.projection.weight",
                     "image_branch.encoder.features.0.weight", "image_branch.norms.3.weight"):
            assert float(dict(model.named_parameters())[name].grad.abs().sum()) > 0, (head, name)


def test_b12_whole_model_overfits_a_tiny_batch():
    """Embedding -> MRTE / GAT -> C2F rounds -> heads -> loss -> Adam, end to end.  Three samples, lr 3e-3 instead of the
    paper's 1e-4 so that a few hundred steps suffice.  Sanity check done without autograd: Adam on the direct head alone
    (z_T taken from this model, NumPy gradients) reaches 23 % of the initial loss with these settings, so the threshold
    below leaves a margin; training the front end as well can only help the fit."""
    need_autograd()
    from omnifuser.models import multitask_loss

    torch.manual_seed(0)
    model = build_full(seed=90, prediction_head="direct", num_rtd_layers=2, num_gat_layers=1)
    model.train()
    inputs = make_inputs(91, batch=3, n_images=2)
    rng = np.random.default_rng(92)
    force_target, class_target = t6.as_tensor(rng.standard_normal((3, HORIZON, CHANNELS))), torch.as_tensor(rng.integers(0, CLASSES, (3, HORIZON)))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(300):
        optimizer.zero_grad()
        out = model(t6.as_tensor(inputs["force"]), t6.as_tensor(inputs["images"]), as_long(inputs["indices"]),
                    as_long(inputs["cycle"]), as_long(inputs["batch"]), as_long(inputs["relative"]))
        loss = multitask_loss(out, force_target, class_target)[0]
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert losses[-1] < 0.4 * losses[0], (losses[0], losses[-1])


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    failed = skipped = 0
    for name in names:
        try:
            globals()[name]()
            print(f"PASS  {name}")
        except unittest.SkipTest as reason:
            skipped += 1
            print(f"SKIP  {name}  ({reason})")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(names) - failed - skipped} passed, {skipped} skipped, {failed} failed")
    sys.exit(1 if failed else 0)
