"""Saving a trained fold as a portable model file and reusing it without training."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_alignment_patch import make_nonastreda

from omnifuser.cli import main
from omnifuser.model_bundle import (
    BUNDLE_FORMAT,
    export_model,
    find_checkpoint,
    load_model,
    predict,
)
from omnifuser.spectrogram import SpectrogramSettings
from omnifuser.train import train_fold

TINY = {"num_rtd_layers": 2, "num_gat_layers": 1, "image_height": 32, "image_width": 96}


def _train(root: Path, output: Path, **overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "data_dir": root, "test_tool": "T1", "output_dir": output, "epochs": 1, "batch_size": 2,
        "input_len": 24, "horizon": 12, "stride": 12, "device_name": "cpu", "trim_pass": False,
        "pool": 4, "image_encoder": "compact", "max_train_windows": 6, "num_workers": 0,
        "config_overrides": TINY,
    }
    settings.update(overrides)
    return train_fold(**settings)


def _same_scores(predicted: dict[str, Any], trained: dict[str, Any]) -> None:
    for key in ("loss", "force_loss", "class_loss", "classification_accuracy", "force_mse_normalized"):
        assert predicted[key] == pytest.approx(trained[key], rel=1e-5, abs=1e-6), key


@pytest.fixture(scope="module")
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_nonastreda(tmp_path_factory.mktemp("bundle") / "data", tools=(1, 2, 3), runs=(1,), length=120)


def test_every_finished_fold_saves_its_final_model_as_a_reusable_file(data_root: Path, tmp_path: Path) -> None:
    result = _train(data_root, tmp_path / "run")
    fold_file = tmp_path / "run" / "fold-T1.pt"
    assert torch.load(fold_file, weights_only=True)["format"] == BUNDLE_FORMAT
    # usable straight from the run folder, no export step needed
    _same_scores(predict(fold_file, data_root, "T1", tmp_path / "direct"), result["test"])


def test_legacy_fold_checkpoints_are_converted_by_export(data_root: Path, tmp_path: Path) -> None:
    result = _train(data_root, tmp_path / "run")
    bundle = torch.load(tmp_path / "run" / "fold-T1.pt", weights_only=True)
    legacy = {  # the fold-Tn.pt layout written before 2026-10-26
        "model_state": bundle["model_state"], "config": bundle["config"],
        "normalizer_mean": np.asarray(bundle["normalizer"]["mean"], np.float32),
        "normalizer_scale": np.asarray(bundle["normalizer"]["scale"], np.float32), "result": result,
    }
    torch.save(legacy, tmp_path / "old-fold-T1.pt")
    with pytest.raises(ValueError, match="convert an older fold checkpoint"):
        load_model(tmp_path / "old-fold-T1.pt")
    export_model(tmp_path / "old-fold-T1.pt", tmp_path / "converted.pt")
    _same_scores(predict(tmp_path / "converted.pt", data_root, "T1", tmp_path / "out"), result["test"])


def test_exported_model_reproduces_the_fold_test_scores_without_training(data_root: Path, tmp_path: Path) -> None:
    result = _train(data_root, tmp_path / "run")
    summary = export_model(tmp_path / "run", tmp_path / "model.pt")
    assert summary["test_tool"] == "T1" and summary["trained_on"] == result["train_tools"]

    bundle = torch.load(tmp_path / "model.pt", weights_only=True)  # the safe loader is enough
    assert bundle["format"] == BUNDLE_FORMAT
    assert bundle["data"]["pool"] == 4 and bundle["data"]["evaluation_stride"] == result["evaluation_stride"]
    loaded = load_model(tmp_path / "model.pt")
    assert not loaded.model.training
    assert np.allclose(loaded.normalizer.scale, result["normalizer_scale"])

    metrics = predict(tmp_path / "model.pt", data_root, "T1", tmp_path / "predictions")
    _same_scores(metrics, result["test"])
    assert metrics["prediction"]["tool_role"] == "test" and "warning" not in metrics["prediction"]
    with np.load(tmp_path / "predictions" / "predictions-T1.npz") as arrays:
        assert arrays["force_predictions"].shape[1:] == (12, 3)
    assert json.loads((tmp_path / "predictions" / "metrics-T1.json").read_text(encoding="utf-8"))["loss"] == metrics["loss"]

    # another tool works, with a warning when the model trained on it
    train_tool = result["train_tools"][0]
    other = predict(tmp_path / "model.pt", data_root, train_tool, tmp_path / "predictions", drop_modality="image")
    assert other["prediction"]["tool_role"] == "train" and "warning" in other["prediction"]
    assert (tmp_path / "predictions" / f"predictions-{train_tool}-image_missing.npz").is_file()


def test_frozen_pretrained_model_loads_without_downloading_weights(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torchvision.models as tvm

    real = tvm.efficientnet_b0
    monkeypatch.setattr(tvm, "efficientnet_b0", lambda weights=None: real(weights=None))
    result = _train(
        data_root, tmp_path / "run", image_encoder="efficientnet_b0",
        config_overrides={**TINY, "pretrained": True, "freeze_image_encoder": True},
    )
    export_model(tmp_path / "run" / "fold-T1.pt", tmp_path / "model.pt")

    def no_download(weights: Any = None) -> torch.nn.Module:
        assert weights is None, "loading a saved model must not download ImageNet weights"
        return real(weights=None)

    monkeypatch.setattr(tvm, "efficientnet_b0", no_download)
    loaded = load_model(tmp_path / "model.pt")
    assert loaded.config.pretrained and loaded.config.freeze_image_encoder
    _same_scores(predict(tmp_path / "model.pt", data_root, "T1", tmp_path / "predictions"), result["test"])


def test_spectrogram_model_is_reused_with_its_own_image_settings(data_root: Path, tmp_path: Path) -> None:
    settings = SpectrogramSettings(window=64, hop=16)
    result = _train(data_root, tmp_path / "run", spectrogram=settings)
    export_model(tmp_path / "run", tmp_path / "model.pt")
    loaded = load_model(tmp_path / "model.pt")
    assert loaded.spectrogram == settings and loaded.config.image_channels == 7
    _same_scores(predict(tmp_path / "model.pt", data_root, "T1", tmp_path / "predictions"), result["test"])


def test_export_needs_a_fold_when_a_run_holds_several_and_load_checks_the_format(
    data_root: Path, tmp_path: Path
) -> None:
    _train(data_root, tmp_path / "run")
    shutil.copy(tmp_path / "run" / "fold-T1.pt", tmp_path / "run" / "fold-T2.pt")
    with pytest.raises(ValueError, match="choose one with --fold"):
        find_checkpoint(tmp_path / "run")
    assert find_checkpoint(tmp_path / "run", "T2").name == "fold-T2.pt"
    with pytest.raises(ValueError, match="not in"):
        find_checkpoint(tmp_path / "run", "T9")
    torch.save({"format": "something-else"}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not an OmniFuser model file"):
        load_model(tmp_path / "other.pt")


def test_export_and_predict_commands(data_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _train(data_root, tmp_path / "run")
    capsys.readouterr()  # drop the training run's progress lines
    main(["export", str(tmp_path / "run"), "--fold", "T1", "--output", str(tmp_path / "models" / "m.pt")])
    assert json.loads(capsys.readouterr().out)["test_tool"] == "T1"
    main(["predict", "--model", str(tmp_path / "models" / "m.pt"), "--data-dir", str(data_root),
          "--tool", "T1", "--output-dir", str(tmp_path / "out"), "--max-windows", "2"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["windows"] == 2 and printed["tool_role"] == "test"
