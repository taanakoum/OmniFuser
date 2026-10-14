"""Regression tests for the integrated alignment features.

    python -m pytest tests/test_alignment_patch.py        # or:  python tests/test_alignment_patch.py

Tests marked (torch) are skipped when PyTorch is not installed. The others need only numpy, scipy and Pillow.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import traceback
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from PIL import Image
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch

    HAVE_TORCH = hasattr(torch, "nn")
except ImportError:  # pragma: no cover
    HAVE_TORCH = False


def need_torch() -> None:
    if not HAVE_TORCH:
        raise unittest.SkipTest("PyTorch is not installed")


CLASS = ("sharp", "used", "dulled")


def make_nonastreda(root, tools=(1, 2, 3), runs=(1, 2), length=700, seed=0, dup_b12=None):
    """Tiny Nonastreda-shaped folder: labels.csv, labels_reg.csv, forces_xyz_raw.mat, tool/*.jpg.

    B4 carries the identical force of B3 in every run (as in the real file); ``dup_b12`` = (tool, run) makes
    B2 identical to B1 as well.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "tool").mkdir(parents=True, exist_ok=True)
    ids, base = [], []
    csv_rows, reg_rows = ["id,image_label,tool_label"], ["id,gaps,flank_wear,overhang"]
    for tool in tools:
        for run in runs:
            made = {}
            for block in (1, 2, 3, 4):
                rid = f"T{tool}R{run}B{block}"
                if block == 4:
                    force = made[3]
                elif block == 2 and dup_b12 == (tool, run):
                    force = made[1]
                else:
                    level = np.array([100.0, -10.0, -20.0])[:, None]
                    force = level + rng.normal(0, 5, (3, length + 10 * block))
                made[block] = force
                label = (run + block) % 3
                csv_rows.append(f"{rid},{CLASS[label]},{label + 1}")
                reg_rows.append(f"{rid},1.0,{10 * block + 5 * run + tool + 0.5 * (block == 4)},2.0")
                Image.fromarray(rng.integers(0, 255, (60, 200, 3), dtype=np.uint8)).save(
                    root / "tool" / f"{rid}.jpg", "JPEG"
                )
                row = np.empty(4, dtype=object)
                row[0] = np.array([f"{rid}.jpg"])
                row[1] = np.array([CLASS[label]])
                row[2] = np.array([str(label + 1)])
                row[3] = force.astype(np.float64)
                ids.append(rid)
                base.append(row)
    (root / "labels.csv").write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    (root / "labels_reg.csv").write_text("\n".join(reg_rows) + "\n", encoding="utf-8")
    array = np.empty((len(base), 4), dtype=object)
    for i, row in enumerate(base):
        array[i, :] = row
    savemat(root / "forces_xyz_raw.mat", {"baseDatastore": array})
    return root


class Workspace:
    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="omnifuser_test_"))
        return self.path

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


# ------------------------------------------------------------------------------------------- config (no torch)
def test_config_accepts_new_options_and_rejects_bad_ones():
    from omnifuser.config import ModelConfig

    config = ModelConfig(prediction_head="direct", als_mode="k_colnorm", modality_dropout=0.2, pre_attn=True)
    assert config.prediction_head == "direct" and config.pre_attn
    assert ModelConfig().pre_attn is False and ModelConfig().als_mode == "k_colnorm"
    bad_options: tuple[dict[str, Any], ...] = (
        {"als_mode": "x"}, {"modality_dropout": 1.0}, {"modality_dropout": -0.1}, {"prediction_head": "x"}
    )
    for bad in bad_options:
        try:
            ModelConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be rejected")


# ------------------------------------------------------------------------------------------- data (no torch)
def test_block_mean_and_pool_records():
    from omnifuser.data import Record, _block_mean, pool_records

    force = np.arange(30, dtype=np.float32).reshape(10, 3)
    pooled = _block_mean(force, 4)
    assert pooled.shape == (2, 3) and np.allclose(pooled[0], force[:4].mean(0)) and np.allclose(pooled[1], force[4:8].mean(0))
    assert _block_mean(force, 1) is force
    record = Record("T1R1B1", "T1", 1, 1, force, 0, Path("x.jpg"))
    assert pool_records([record], 5)[0].force.shape == (2, 3)
    try:
        _block_mean(force, 11)
    except ValueError:
        pass
    else:
        raise AssertionError("pool larger than the record must fail")


def test_loader_pool_follows_trim_and_dedup_covers_all_blocks():
    from omnifuser.data import load_nonastreda

    with Workspace() as tmp:
        make_nonastreda(tmp, tools=(1,), runs=(1, 2), dup_b12=(1, 1))
        plain = load_nonastreda(tmp, trim_pass=False)
        ids = sorted(r.record_id for r in plain)
        # run 1: B1=B2 and B3=B4 -> 2 records; run 2: B3=B4 -> 3 records. The record with the larger flank wear stays
        # (B2 beats B1 and B4 beats B3 in this fixture).
        assert ids == ["T1R1B2", "T1R1B4", "T1R2B1", "T1R2B2", "T1R2B4"], ids
        kept = {r.record_id: r for r in plain}
        assert kept["T1R1B4"].label == (1 + 4) % 3 and kept["T1R1B2"].label == (1 + 2) % 3
        pooled = load_nonastreda(tmp, trim_pass=False, pool=10)
        for a, b in zip(plain, pooled):
            assert len(b.force) == len(a.force) // 10
        trimmed = load_nonastreda(tmp, trim_pass=True, pool=10)
        assert all(len(t.force) <= len(p.force) for t, p in zip(trimmed, pooled))


def test_image_lag_uses_previous_pass_photo_but_own_label():
    from omnifuser.data import load_nonastreda, make_tool_timelines

    with Workspace() as tmp:
        make_nonastreda(tmp, tools=(1,), runs=(1, 2))
        records = load_nonastreda(tmp, trim_pass=False)
        own = make_tool_timelines(records, image_lag=0)[0]
        lag = make_tool_timelines(records, image_lag=1)[0]
        assert own.image_labels is not None and np.array_equal(own.image_labels, own.labels)
        assert np.array_equal(lag.labels, own.labels)
        start = 0
        previous = None
        for record in sorted(records, key=lambda r: (r.run, r.block)):
            segment = slice(start, start + len(record.force))
            shown_path = lag.image_paths[int(lag.image_codes[segment][0])]
            assert (lag.image_codes[segment] == lag.image_codes[segment][0]).all()
            if previous is None:
                assert shown_path == record.image_path
            else:
                assert shown_path == previous.image_path
                assert lag.image_labels is not None
                assert (lag.image_labels[segment] == previous.label).all()
            assert own.image_paths[int(own.image_codes[segment][0])] == record.image_path
            previous, start = record, start + len(record.force)
        try:
            make_tool_timelines(records, image_lag=2)
        except ValueError:
            pass
        else:
            raise AssertionError("image_lag 2 must be rejected")


def test_window_item_exposes_label_of_the_shown_photo():
    from omnifuser.data import (
        ForceNormalizer,
        WindowDataset,
        load_nonastreda,
        make_tool_timelines,
    )

    with Workspace() as tmp:
        make_nonastreda(tmp, tools=(1,), runs=(1,), length=300)
        records = load_nonastreda(tmp, trim_pass=False)
        for lag in (0, 1):
            timelines = make_tool_timelines(records, image_lag=lag)
            data = WindowDataset(timelines, ForceNormalizer.fit(timelines), 96, 24, 120)
            for index in range(len(data)):
                item = data[index]
                shown = timelines[0].image_labels
                assert shown is not None
                expected = shown[index * 120 + 95]
                assert int(item["shown_label"]) == int(expected)
        # a ToolTimeline built without image_labels (old call sites) still works
        import dataclasses

        legacy = dataclasses.replace(timelines[0], image_labels=None)
        item = WindowDataset([legacy], ForceNormalizer.fit([legacy]), 96, 24, 120)[0]
        assert int(item["shown_label"]) == int(legacy.labels[95])


def test_image_bank_pixels_match_the_reference_decode():
    from omnifuser.data import ImageBank

    with Workspace() as tmp:
        make_nonastreda(tmp, tools=(1,), runs=(1,), length=200)
        paths = sorted((tmp / "tool").glob("*.jpg"))
        bank = ImageBank(paths + paths[:1], height=32, width=96)
        assert len(bank) == len(paths) and bank.pixels.shape == (len(paths), 32, 96, 3)
        for position, path in enumerate(paths):
            with Image.open(path) as source:
                reference = np.asarray(source.convert("RGB").resize((96, 32), Image.Resampling.BILINEAR))
            assert np.array_equal(bank.pixels[bank.index[path]], reference)


# --------------------------------------------------------------------------------------- evaluation (no torch)
def test_cumulative_horizon_report_and_baselines():
    from omnifuser.evaluation import baseline_report, cumulative_horizon_report

    rng = np.random.default_rng(1)
    n, p = 40, 24
    target = rng.normal(size=(n, p, 3)).astype(np.float32)
    classes = rng.integers(0, 3, (n, p))
    probabilities = np.eye(3)[classes]
    perfect = cumulative_horizon_report(target, target, probabilities, classes)
    assert sorted(perfect) == ["12", "24"], perfect.keys()
    assert perfect["12"]["mse"] == 0.0 and perfect["24"]["accuracy"] == 1.0 and perfect["24"]["f1_macro"] == 1.0
    shifted = cumulative_horizon_report(target + 0.5, target, probabilities, classes)
    assert abs(shifted["12"]["mse"] - 0.25) < 1e-6 and abs(shifted["12"]["mae"] - 0.5) < 1e-6
    wrong = cumulative_horizon_report(target, target, np.roll(probabilities, 1, axis=-1), classes)
    assert wrong["24"]["accuracy"] == 0.0
    last = rng.normal(size=(n, 3)).astype(np.float32)
    shown = classes[:, 0]
    base = baseline_report(last, np.repeat(last[:, None, :], p, axis=1), shown, np.repeat(shown[:, None], p, axis=1))
    assert base["last_value_mse"] == 0.0 and base["persistence_class_accuracy"] == 1.0
    assert abs(base["zero_prediction_mse"] - float(np.square(last).mean())) < 1e-6


def test_regression_report_has_normalised_per_horizon_metrics():
    from omnifuser.evaluation import regression_report

    pred = np.zeros((5, 24, 3), np.float32)
    target = np.ones((5, 24, 3), np.float32)
    report = regression_report(pred, target, np.zeros(3, np.float32), np.ones(3, np.float32))
    assert len(report["by_horizon_normalized"]) == 24 and report["by_horizon_normalized"][0]["mse"] == 1.0


def test_diagnostics_window_starts():
    spec = importlib.util.spec_from_file_location("task_diagnostics", ROOT / "scripts" / "task_diagnostics.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    starts = module.window_starts(1000, 96, 24, 24, None)
    assert starts[0] == 0 and starts[-1] + 96 + 24 <= 1000 and np.all(np.diff(starts) == 24)
    assert len(module.window_starts(100000, 96, 24, 24, 100)) <= 100


# --------------------------------------------------------------------------------------------- models (torch)
def _tiny_inputs(batch=2, config=None):
    import torch

    steps = 96 if config is None else config.input_len
    force = torch.randn(batch, steps, 3)
    images = torch.randn(batch, 1, 3, 128, 384)
    indices = torch.zeros(batch, steps, dtype=torch.long)
    return force, images, indices


def _compact(**kwargs):
    from omnifuser.config import ModelConfig

    return ModelConfig(image_encoder="compact", **kwargs)


def test_forward_shapes_for_every_head_and_literal_repeats(tmp=None):
    need_torch()
    import torch

    from omnifuser.models import OmniFuser

    for head in ("paper_literal", "lead_conditioned", "direct"):
        torch.manual_seed(0)
        model = OmniFuser(_compact(prediction_head=head)).eval()
        out = model(*_tiny_inputs())
        assert out["force"].shape == (2, 24, 3) and out["class_logits"].shape == (2, 24, 3), head
        spread = (out["force"] - out["force"][:, :1]).abs().max().detach().item()
        if head == "paper_literal":
            assert spread == 0.0
        else:
            assert spread > 0.0, head


def test_pre_attn_flag_removes_the_extra_attention():
    need_torch()
    from omnifuser.models import OmniFuser

    off, on = OmniFuser(_compact(pre_attn=False)), OmniFuser(_compact(pre_attn=True))
    assert not hasattr(off.fusion_layers[0], "self_r") and hasattr(on.fusion_layers[0], "self_r")
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(on) > count(off)


def test_als_modes_normalise_as_documented():
    need_torch()
    import torch

    from omnifuser.models import CrossModalFusionLayer

    torch.manual_seed(0)
    batch, steps, d, k = 3, 96, 32, 4
    x = torch.ones(batch, steps, d)
    results = {}
    for mode in ("time", "k_sum", "k_colnorm"):
        layer = CrossModalFusionLayer(_compact(als_mode=mode))
        identity = torch.nn.Identity()
        landmarks, _ = layer._landmarks(x, layer.score_r, identity)
        results[mode] = landmarks
    assert torch.allclose(results["time"], torch.ones(batch, k, d), atol=1e-5)  # each landmark = weighted average of ones
    assert torch.allclose(results["k_colnorm"], torch.ones(batch, k, d), atol=1e-5)
    assert torch.allclose(results["k_sum"].sum(dim=1), torch.full((batch, d), float(steps)), atol=1e-3)  # token mass T


def test_modality_masking():
    need_torch()
    import torch

    from omnifuser.models import OmniFuser

    torch.manual_seed(0)
    model = OmniFuser(_compact(modality_dropout=0.99))
    sensor, image = torch.ones(4000, 2, 3), torch.ones(4000, 2, 3)
    model.train()
    s, i = model._mask_modalities(sensor, image, None)
    s_zero, i_zero = (s.sum((1, 2)) == 0), (i.sum((1, 2)) == 0)
    assert not bool((s_zero & i_zero).any())  # never both
    assert abs(float(s_zero.float().mean()) - 0.495) < 0.04 and abs(float(i_zero.float().mean()) - 0.495) < 0.04
    model.eval()
    s, i = model._mask_modalities(sensor, image, None)
    assert torch.equal(s, sensor) and torch.equal(i, image)
    s, i = model._mask_modalities(sensor, image, "image")
    assert torch.equal(s, sensor) and float(i.abs().sum()) == 0.0
    try:
        model._mask_modalities(sensor, image, "audio")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown modality must be rejected")
    inputs = _tiny_inputs()
    assert not torch.allclose(model(*inputs)["force"], model(*inputs, drop_modality="sensor")["force"])


def test_freeze_bn_keeps_batchnorm_in_eval_mode():
    need_torch()
    import torch

    from omnifuser.models import OmniFuser

    for freeze in (True, False):
        model = OmniFuser(_compact(freeze_bn=freeze))
        model.train()
        norms = [m for m in model.image_branch.encoder.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        assert norms and all(m.training == (not freeze) for m in norms)
        assert model.temporal_embedding.training  # everything else still trains


def test_backward_gives_finite_gradients_for_every_option():
    need_torch()
    import torch

    from omnifuser.models import OmniFuser, multitask_loss

    options = [
        {"prediction_head": "direct"},
        {"prediction_head": "paper_literal", "pre_attn": True},
        {"prediction_head": "lead_conditioned", "als_mode": "k_colnorm", "modality_dropout": 0.3},
        {"prediction_head": "direct", "als_mode": "k_sum", "freeze_bn": True},
    ]
    for option in options:
        torch.manual_seed(0)
        model = OmniFuser(_compact(**option)).train()
        out = model(*_tiny_inputs(4))
        loss, _ = multitask_loss(out, torch.randn(4, 24, 3), torch.randint(0, 3, (4, 24)))
        loss.backward()
        assert torch.isfinite(loss), option
        bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not bad, (option, bad)
        if option.get("prediction_head") == "direct":
            assert model.direct_force.weight.grad is not None


def test_bank_tensor_equals_image_tensor_and_collate_carries_shown_label():
    need_torch()
    import torch

    from omnifuser.data import (
        ForceNormalizer,
        ImageBank,
        WindowDataset,
        collate_windows,
        image_tensor,
        load_nonastreda,
        make_tool_timelines,
    )

    with Workspace() as tmp:
        make_nonastreda(tmp, tools=(1,), runs=(1,), length=300)
        records = load_nonastreda(tmp, trim_pass=False)
        timelines = make_tool_timelines(records, image_lag=1)
        bank = ImageBank.from_timelines(timelines, 32, 96)
        for path in timelines[0].image_paths:
            assert torch.equal(bank.tensor(path), image_tensor(path, 32, 96))
        data = WindowDataset(timelines, ForceNormalizer.fit(timelines), 96, 24, 120)
        samples = [data[0], data[1]]
        with_bank = collate_windows(samples, 32, 96, bank=bank)
        without = collate_windows(samples, 32, 96)
        assert torch.equal(with_bank["images"], without["images"])
        assert with_bank["shown_label"].shape == (2,) and with_bank["shown_label"].dtype == torch.int64


def _smoke_kwargs(root: Path, out: Path) -> dict[str, Any]:
    return {"data_dir": root, "output_dir": out, "epochs": 2, "batch_size": 2, "device_name": "cpu",
            "trim_pass": False, "image_encoder": "compact", "max_train_windows": 4, "max_eval_windows": 2,
            "num_workers": 0}


def test_train_fold_end_to_end_with_new_options_and_resume():
    need_torch()
    from omnifuser import train as train_module
    from omnifuser.cross_validation import _load_finished_fold
    from omnifuser.train import train_fold

    with Workspace() as tmp:
        root = make_nonastreda(tmp / "data", tools=(1, 2, 3), runs=(1, 2, 3), length=400)
        out = tmp / "out"
        overrides = {"pre_attn": False, "als_mode": "k_colnorm", "modality_dropout": 0.2}
        result = train_fold(test_tool="T1", prediction_head="direct", pool=2, image_lag=1,
                            accumulation_steps=2, config_overrides=overrides, **_smoke_kwargs(root, out))
        assert result["best_validation_epoch"] in (1, 2) and result["selected_epoch"] == result["best_validation_epoch"]
        assert result["pool"] == 2 and result["image_lag"] == 1 and result["training"]["effective_batch_size"] == 4
        assert set(result["test"]["paper_horizons"]) == {"12", "24"}
        assert {"last_value_mse", "zero_prediction_mse", "persistence_class_accuracy"} <= set(result["test"]["baselines"])
        assert result["config"]["als_mode"] == "k_colnorm" and result["config"]["prediction_head"] == "direct"
        assert not (out / "fold-T1-resume.pt").exists()
        assert json.loads((out / "fold-T1.json").read_text())["selected_epoch"] == result["selected_epoch"]
        # a finished fold is reused only when every setting matches
        from dataclasses import asdict

        from omnifuser.config import ModelConfig

        expected = asdict(ModelConfig(image_encoder="compact", prediction_head="direct", **overrides))
        settings = {"pool": 2, "image_lag": 1, "decimation": 1, "label_column": "image_label",
                    "dedup_identical_forces": True, "trim_pass": False, "model_selection": "best_validation_loss"}
        assert _load_finished_fold(out, "T1", expected, settings, 2, 2) is not None
        assert _load_finished_fold(out, "T1", expected, {**settings, "pool": 3}, 2, 2) is None
        # interrupt after the first epoch, then resume
        out2 = tmp / "out2"
        real = train_module._run_epoch
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 3:  # third call = training pass of epoch 2
                raise KeyboardInterrupt
            return real(*args, **kwargs)

        with mock.patch.object(train_module, "_run_epoch", flaky):
            try:
                train_fold(test_tool="T2", prediction_head="direct", **_smoke_kwargs(root, out2))
            except KeyboardInterrupt:
                pass
        assert (out2 / "fold-T2-resume.pt").exists()
        resumed = train_fold(test_tool="T2", prediction_head="direct", **_smoke_kwargs(root, out2))
        assert [h["epoch"] for h in resumed["history"]] == [1, 2]
        assert not (out2 / "fold-T2-resume.pt").exists()


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
        except Exception:  # noqa: BLE001 - standalone runner reports every failure and continues
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(names) - failed - skipped} passed, {skipped} skipped, {failed} failed")
    sys.exit(1 if failed else 0)
