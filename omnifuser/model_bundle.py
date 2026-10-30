"""Trained OmniFuser models as portable files, reused without training.

Every finished fold saves its final model as ``fold-Tn.pt`` in this format (``train_fold``
calls ``make_bundle``). The file holds everything needed to use the model elsewhere:

* the weights, including the frozen ImageNet backbone, so no download is needed;
* the model configuration;
* the force normalizer fitted on the fold's training tools;
* the data settings that recreate its inputs (pooling, image lag, windows, spectrograms);
* the fold's tools, selected epoch and test scores.

``load_model`` reads it with PyTorch's safe loader (``weights_only=True``), ``predict``
forecasts a tool's windows with it, and ``export_model`` copies one fold's model out
under a chosen name (it also converts checkpoints written before this format existed).

A fold's model was trained on 8 tools and selected on a 9th; its scores are test scores
only on the fold's held-out tool. ``predict`` warns when asked for another tool.
"""
from __future__ import annotations

import json
import os
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import omnifuser
from omnifuser.config import ModelConfig
from omnifuser.data import ForceNormalizer, collate_windows
from omnifuser.models import OmniFuser
from omnifuser.spectrogram import SpectrogramSettings

BUNDLE_FORMAT = "omnifuser-model/1"
# Data settings a model needs to see its inputs as in training (keys of the fold result).
DATA_KEYS = (
    "label_column", "dedup_identical_forces", "eval_dedup_identical_forces", "trim_pass",
    "decimation", "pool", "image_lag", "stride", "evaluation_stride",
)
TRAINING_KEYS = ("test_tool", "validation_tool", "train_tools", "selected_epoch", "model_selection")
SUMMARY_METRICS = ("loss", "force_loss", "class_loss", "classification_accuracy",
                   "force_mse_normalized", "force_mae_normalized")


@dataclass(frozen=True)
class LoadedModel:
    """A model ready to use, with what it needs to prepare its inputs."""

    model: OmniFuser
    config: ModelConfig
    normalizer: ForceNormalizer
    data: dict[str, Any]
    training: dict[str, Any]
    spectrogram: SpectrogramSettings | None


def make_bundle(
    config: Mapping[str, Any],
    model_state: Mapping[str, torch.Tensor],
    normalizer_mean: Sequence[float] | np.ndarray,
    normalizer_scale: Sequence[float] | np.ndarray,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """The portable model file's content, from a fold's model and its result dict."""
    data = {key: result[key] for key in DATA_KEYS if key in result}
    data["spectrogram"] = result.get("spectrogram")
    return {
        "format": BUNDLE_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "omnifuser_version": omnifuser.__version__,
        "config": dict(config),
        "model_state": {key: value.detach().cpu() for key, value in model_state.items()},
        "normalizer": {
            "mean": [float(value) for value in normalizer_mean],
            "scale": [float(value) for value in normalizer_scale],
        },
        "data": data,
        "training": {
            **{key: result[key] for key in TRAINING_KEYS if key in result},
            **{key: result["training"][key] for key in ("epochs", "learning_rate", "batch_size", "seed")},
        },
        "test_metrics": {key: float(result["test"][key]) for key in SUMMARY_METRICS if key in result["test"]},
    }


def find_checkpoint(source: str | Path, fold: str | None = None) -> Path:
    """A fold's model file: ``source`` itself, or ``fold-<fold>.pt`` in a run folder."""
    path = Path(source)
    if path.is_file():
        return path
    available = sorted(
        (p for p in path.glob("fold-T*.pt") if not p.name.endswith("-resume.pt")),
        key=lambda p: int(p.stem.split("-T")[1]),
    )
    names = [p.stem.removeprefix("fold-") for p in available]
    if not available:
        raise FileNotFoundError(f"no finished fold (fold-Tn.pt) in {path}")
    if fold is None:
        if len(available) > 1:
            raise ValueError(f"{path} holds {len(available)} folds; choose one with --fold ({', '.join(names)})")
        return available[0]
    if fold not in names:
        raise ValueError(f"fold {fold!r} not in {path}; available: {', '.join(names)}")
    return path / f"fold-{fold}.pt"


def _read_bundle(checkpoint_path: Path) -> dict[str, Any]:
    try:
        bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError):  # older checkpoints hold NumPy arrays, refused here
        bundle = None
    if isinstance(bundle, dict) and bundle.get("format") == BUNDLE_FORMAT:
        return bundle
    # Written by train_fold before 2026-10-26: our own file, so the full loader is acceptable.
    legacy = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(legacy, dict) or not {"model_state", "config", "result"} <= set(legacy):
        raise ValueError(f"{checkpoint_path} is neither an OmniFuser model file nor a fold checkpoint")
    return make_bundle(
        legacy["config"], legacy["model_state"], legacy["normalizer_mean"],
        legacy["normalizer_scale"], legacy["result"],
    )


def export_model(source: str | Path, output: str | Path, fold: str | None = None) -> dict[str, Any]:
    """Copy one fold's model to ``output`` as a model file; returns a short description."""
    checkpoint_path = find_checkpoint(source, fold)
    bundle = _read_bundle(checkpoint_path)
    bundle["source_checkpoint"] = str(checkpoint_path)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    torch.save(bundle, temporary)
    os.replace(temporary, output_path)
    return {
        "model_file": str(output_path),
        "size_MB": round(output_path.stat().st_size / 2**20, 1),
        "trained_on": bundle["training"].get("train_tools"),
        "test_tool": bundle["training"].get("test_tool"),
        "test_metrics": bundle["test_metrics"],
    }


def load_model(path: str | Path, device: str | torch.device = "cpu") -> LoadedModel:
    """Load a model file (a finished ``fold-Tn.pt`` or an export), in eval mode on ``device``."""
    try:
        bundle = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError):
        bundle = None  # not loadable safely, e.g. a fold checkpoint from before 2026-10-26
    if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
        raise ValueError(
            f"{path} is not an OmniFuser model file ({BUNDLE_FORMAT}); "
            "convert an older fold checkpoint with 'omnifuser export'"
        )
    config = ModelConfig(**bundle["config"])
    model = OmniFuser(config, load_pretrained=False)
    model.load_state_dict(bundle["model_state"])
    model.to(device).eval()
    normalizer = ForceNormalizer(
        np.asarray(bundle["normalizer"]["mean"], dtype=np.float32),
        np.asarray(bundle["normalizer"]["scale"], dtype=np.float32),
    )
    spectrogram = bundle["data"].get("spectrogram")
    return LoadedModel(
        model=model,
        config=config,
        normalizer=normalizer,
        data=bundle["data"],
        training=bundle["training"],
        spectrogram=SpectrogramSettings(**spectrogram) if spectrogram else None,
    )


def predict(
    model_path: str | Path,
    data_dir: str | Path,
    tool: str,
    output_dir: str | Path,
    *,
    device_name: str = "cpu",
    batch_size: int = 32,
    drop_modality: str | None = None,
    max_windows: int | None = None,
    num_workers: int = 0,
    evaluation_stride: int | None = None,
) -> dict[str, Any]:
    """Forecast every evaluation window of ``tool`` with a saved model, without training.

    Writes ``predictions-<tool>[-<mode>].npz`` (forecasts and targets in newtons, class
    probabilities and targets) and ``metrics-<tool>[-<mode>].json``; returns the metrics.
    ``evaluation_stride`` replaces the saved window stride (files then end in
    ``-stride<n>``): a stride that is not a multiple of the pass length starts the
    forecasts at every phase of a pass instead of near pass boundaries.
    """
    if evaluation_stride is not None and evaluation_stride < 1:
        raise ValueError("evaluation_stride must be positive")
    # Imported here: train.py itself imports this module to save its fold models.
    from omnifuser.train import (
        _run_epoch,
        image_bank_for,
        load_timelines,
        save_arrays,
        window_dataset,
    )

    device = torch.device(device_name)
    loaded = load_model(model_path, device)
    data = loaded.data
    timelines, _, raw_passes = load_timelines(
        data_dir,
        label_column=data["label_column"],
        decimation=data["decimation"],
        dedup_identical_forces=data["eval_dedup_identical_forces"],
        trim_pass=data["trim_pass"],
        pool=data["pool"],
        image_lag=data["image_lag"],
        spectrogram=loaded.spectrogram,
    )
    by_tool = {timeline.tool: timeline for timeline in timelines}
    if tool not in by_tool:
        raise ValueError(f"unknown tool {tool!r}; choose one of {sorted(by_tool, key=lambda t: int(t[1:]))}")
    stride = evaluation_stride or data["evaluation_stride"]
    dataset: Any = window_dataset(
        [by_tool[tool]], loaded.normalizer, loaded.config, stride,
        spectrogram=loaded.spectrogram, raw_passes=raw_passes, pool=data["pool"],
    )
    if max_windows is not None and max_windows < len(dataset):
        dataset = Subset(dataset, range(max_windows))
    bank = image_bank_for([by_tool[tool]], loaded.config, loaded.model, device, loaded.spectrogram)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(collate_windows, height=loaded.config.image_height,
                           width=loaded.config.image_width, bank=bank),
    )
    metrics, predictions = _run_epoch(
        loaded.model, loader, None, device, loaded.normalizer,
        collect_predictions=True, drop_modality=drop_modality,
    )
    if predictions is None:
        raise RuntimeError("prediction returned no outputs")
    role = (
        "test" if tool == loaded.training.get("test_tool")
        else "validation" if tool == loaded.training.get("validation_tool")
        else "train" if tool in loaded.training.get("train_tools", [])
        else "unseen"
    )
    metrics["prediction"] = {
        "model_file": str(model_path),
        "tool": tool,
        "tool_role": role,
        "windows": len(dataset),
        "evaluation_stride": stride,
        "drop_modality": drop_modality,
        "config": asdict(loaded.config),
        "data": data,
    }
    if role in {"train", "validation"}:
        metrics["prediction"]["warning"] = (
            f"{tool} was a {role} tool of this model; its scores are not test scores"
        )
    suffix = f"{tool}-{drop_modality}_missing" if drop_modality else tool
    if evaluation_stride is not None:
        suffix += f"-stride{evaluation_stride}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_arrays(output_path / f"predictions-{suffix}.npz", predictions)
    metrics_path = output_path / f"metrics-{suffix}.json"
    temporary = metrics_path.with_name(metrics_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    os.replace(temporary, metrics_path)
    return metrics
