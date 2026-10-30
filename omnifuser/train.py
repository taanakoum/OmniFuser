from __future__ import annotations

import json
import os
import pickle
import platform
import random
import time
from collections.abc import Mapping, Sized
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from omnifuser.config import ModelConfig
from omnifuser.data import (
    NO_IMAGE,
    PASS_TRIM_REFERENCE_FRACTION,
    PASS_TRIM_THRESHOLD_FRACTION,
    PASS_TRIM_WINDOW_SAMPLES,
    ForceNormalizer,
    ImageBank,
    ToolTimeline,
    WindowDataset,
    collate_windows,
    leave_one_tool_out_folds,
    load_nonastreda,
    make_tool_timelines,
    nonastreda_source_signature,
    pool_records,
)
from omnifuser.evaluation import (
    baseline_report,
    classification_report,
    cumulative_horizon_report,
    photo_change_report,
    regression_report,
    run_number_baseline_accuracy,
    softmax_probabilities,
)
from omnifuser.model_bundle import make_bundle
from omnifuser.models import ImageFeatureEncoder, OmniFuser, multitask_loss
from omnifuser.resources import available_cpus, cpu_threads
from omnifuser.spectrogram import (
    IMAGE_CHANNELS,
    ForceWindowDataset,
    SpectrogramSettings,
    raw_passes_by_tool,
)
from omnifuser.tracking import ExperimentTracker


def model_config(
    *,
    d_model: int,
    input_len: int,
    horizon: int,
    image_encoder: str,
    prediction_head: str,
    config_overrides: dict[str, Any] | None = None,
    spectrogram: SpectrogramSettings | None = None,
) -> ModelConfig:
    """The configuration a fold trains with; fold reuse compares against the same."""
    overrides: dict[str, Any] = dict(config_overrides or {})
    if spectrogram is not None:
        overrides["image_channels"] = IMAGE_CHANNELS
    return ModelConfig(
        d_model=d_model,
        input_len=input_len,
        horizon=horizon,
        image_encoder=image_encoder,
        prediction_head=prediction_head,
        **overrides,
    )


class FrozenFeatureBank:
    """Pooled frozen-backbone features of every photo, computed once per fold.

    A frozen backbone (fixed weights, eval mode) gives a photo the same features in
    every batch and epoch, so they are computed once here and the model receives them
    instead of pixels. The model's outputs are those of computing them per batch;
    only the repeated EfficientNet forward passes are saved.
    """

    def __init__(
        self,
        bank: ImageBank,
        encoder: ImageFeatureEncoder,
        device: torch.device,
        batch_size: int = 64,
    ) -> None:
        paths = [NO_IMAGE, *bank.index]
        was_training = encoder.training
        encoder.eval()
        parts: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(paths), batch_size):
                pixels = torch.stack([bank.tensor(path) for path in paths[start : start + batch_size]])
                parts.append(encoder.backbone_features(pixels.to(device)).float().cpu())
        encoder.train(was_training)
        self.features = torch.cat(parts)
        self.index = {path: position for position, path in enumerate(paths)}

    def __len__(self) -> int:
        return len(self.index)

    def tensor(self, path: Path) -> torch.Tensor:
        return self.features[self.index[Path(path)]]


def load_timelines(
    data_dir: str | Path,
    *,
    label_column: str,
    decimation: int,
    dedup_identical_forces: bool,
    trim_pass: bool,
    pool: int,
    image_lag: int,
    spectrogram: SpectrogramSettings | None = None,
) -> tuple[list[ToolTimeline], int, dict[str, Any]]:
    """Tool timelines, the retained pass count, and (for spectrograms) the raw passes.

    Training and ``model_bundle.predict`` both build their inputs here, so a saved model
    sees exactly the preprocessing it was trained with.
    """
    # Spectrograms need the raw passes, so they are loaded unpooled and pooled here;
    # pool_records gives exactly the pooling load_nonastreda would apply.
    records = load_nonastreda(
        data_dir,
        label_column=label_column,
        decimation=decimation,
        dedup_identical_forces=dedup_identical_forces,
        trim_pass=trim_pass,
        pool=1 if spectrogram is not None else pool,
    )
    raw_passes: dict[str, Any] = {}
    if spectrogram is not None:
        raw_passes = raw_passes_by_tool(records, pool)
        records = pool_records(records, pool)
    return make_tool_timelines(records, image_lag=image_lag), len(records), raw_passes


def window_dataset(
    timelines: list[ToolTimeline],
    normalizer: ForceNormalizer,
    config: ModelConfig,
    stride: int,
    *,
    spectrogram: SpectrogramSettings | None = None,
    raw_passes: dict[str, Any] | None = None,
    pool: int = 1,
) -> WindowDataset:
    """Windows over ``timelines``: photo windows, or windows with spectrogram images."""
    if spectrogram is None:
        return WindowDataset(timelines, normalizer, config.input_len, config.horizon, stride)
    return ForceWindowDataset(
        timelines, normalizer, raw_passes or {}, pool, spectrogram, config.image_height,
        config.image_width, config.input_len, config.horizon, stride,
    )


def image_bank_for(
    timelines: list[ToolTimeline],
    config: ModelConfig,
    model: OmniFuser,
    device: torch.device,
    spectrogram: SpectrogramSettings | None = None,
) -> ImageBank | FrozenFeatureBank | None:
    """The photo cache for ``timelines``; frozen-backbone features instead of pixels."""
    # Spectrogram windows carry their own images; only photo runs need a bank. With a
    # frozen backbone, the bank holds each photo's features instead of its pixels.
    if spectrogram is not None:
        return None
    bank = ImageBank.from_timelines(
        timelines, config.image_height, config.image_width, config.image_crop
    )
    if config.freeze_image_encoder:
        return FrozenFeatureBank(bank, model.image_branch.encoder, device)
    return bank


def _temporary_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def save_arrays(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    # Written to a temporary file, then renamed: a job killed mid-write (e.g. at a
    # cluster time limit) never leaves a truncated file under the final name.
    temporary = _temporary_path(path)
    with temporary.open("wb") as file:
        # NumPy's stubs let a ``**`` key bind ``allow_pickle``; every key here is an array name.
        np.savez_compressed(file, **arrays)  # pyright: ignore[reportArgumentType]
    os.replace(temporary, path)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = _temporary_path(path)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_roc_curves(
    path: Path, curves: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> None:
    save_arrays(
        path,
        {
            f"{name}_{coordinate}": values
            for name, curve in curves.items()
            for coordinate, values in zip(("fpr", "tpr", "thresholds"), curve)
        },
    )


def _dataset_size(loader: DataLoader[Any]) -> int:
    return len(cast(Sized, loader.dataset))


def _limit_dataset(dataset: WindowDataset, limit: int | None, seed: int) -> Any:
    if limit is None or limit >= len(dataset):
        return dataset
    if limit < 1:
        raise ValueError("window limits must be positive")
    indices = random.Random(seed).sample(range(len(dataset)), limit)
    return Subset(dataset, indices)


def _accumulation_group_sample_count(
    dataset_size: int, batch_size: int, step: int, accumulation_steps: int
) -> int:
    group_start_batch = ((step - 1) // accumulation_steps) * accumulation_steps
    group_end_batch = min(group_start_batch + accumulation_steps, (dataset_size + batch_size - 1) // batch_size)
    start = group_start_batch * batch_size
    end = min(group_end_batch * batch_size, dataset_size)
    return end - start


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _cuda_memory_metrics(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _run_epoch(
    model: OmniFuser,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    normalizer: ForceNormalizer,
    *,
    collect_predictions: bool = False,
    accumulation_steps: int = 1,
    drop_modality: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    training = optimizer is not None
    model.train(training)
    # Running sums stay on the device and are copied to the CPU once per epoch: a
    # float() or .cpu() per step would stall the GPU queue at every batch.
    horizon, channels = model.config.horizon, model.config.force_channels

    def zeros(*shape: int) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.float64, device=device)

    totals: dict[str, torch.Tensor] = {}
    correct = torch.zeros((), dtype=torch.long, device=device)
    count = 0
    force_elements = 0
    channel_elements = 0
    horizon_elements = 0
    sums = {
        name: zeros(channels) if "channel" in name else zeros(horizon)
        for name in (
            "normalized_abs_by_channel", "normalized_squared_by_channel",
            "physical_abs_by_channel", "physical_squared_by_channel",
            "normalized_abs_by_horizon", "normalized_squared_by_horizon",
            "physical_abs_by_horizon", "physical_squared_by_horizon",
        )
    }
    confusion_logits: list[torch.Tensor] = []
    confusion_targets: list[torch.Tensor] = []
    force_predictions: list[torch.Tensor] = []
    force_targets: list[torch.Tensor] = []
    last_input_forces: list[torch.Tensor] = []
    shown_labels: list[torch.Tensor] = []
    grad_context = torch.enable_grad() if training else torch.no_grad()
    force_scale = torch.as_tensor(normalizer.scale, device=device, dtype=torch.float64)
    with grad_context:
        batches = len(loader)
        if training:
            optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            tensors = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            outputs = model(
                tensors["force"],
                tensors["images"],
                tensors["image_indices"],
                tensors["cycle"],
                tensors["batch"],
                tensors["relative"],
                drop_modality=drop_modality,
            )
            loss, components = multitask_loss(
                outputs, tensors["force_target"], tensors["class_target"]
            )
            if training:
                if loader.batch_size is None:
                    raise ValueError("gradient accumulation requires a fixed DataLoader batch size")
                group_samples = _accumulation_group_sample_count(
                    _dataset_size(loader), loader.batch_size, step, accumulation_steps
                )
                batch_samples = tensors["force"].shape[0]
                (loss * (batch_samples / group_samples)).backward()
                if step % accumulation_steps == 0 or step == batches:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            batch_size = tensors["force"].shape[0]
            for key, value in components.items():
                totals[key] = totals.get(key, zeros()) + value.double() * batch_size
            logits = outputs["class_logits"].detach()
            correct += (logits.argmax(dim=-1) == tensors["class_target"]).sum()
            count += tensors["class_target"].numel()
            confusion_logits.append(logits)
            confusion_targets.append(tensors["class_target"])
            if collect_predictions:
                force_predictions.append(outputs["force"].detach())
                force_targets.append(tensors["force_target"])
                last_input_forces.append(tensors["force"][:, -1])
                shown_labels.append(tensors["shown_label"])
            difference = (outputs["force"].detach() - tensors["force_target"]).double()
            physical_difference = difference * force_scale[None, None, :]
            for kind, values in (("normalized", difference), ("physical", physical_difference)):
                sums[f"{kind}_abs_by_channel"] += values.abs().sum(dim=(0, 1))
                sums[f"{kind}_squared_by_channel"] += values.square().sum(dim=(0, 1))
                sums[f"{kind}_abs_by_horizon"] += values.abs().sum(dim=(0, 2))
                sums[f"{kind}_squared_by_horizon"] += values.square().sum(dim=(0, 2))
            force_elements += difference.numel()
            channel_elements += batch_size * horizon
            horizon_elements += batch_size * channels
    # One copy to the CPU per epoch.
    totals_cpu = {key: float(value) for key, value in totals.items()}
    correct_count = int(correct)
    host = {name: value.cpu().numpy() for name, value in sums.items()}
    normalized_abs_by_channel = host["normalized_abs_by_channel"]
    normalized_squared_by_channel = host["normalized_squared_by_channel"]
    physical_abs_by_channel = host["physical_abs_by_channel"]
    physical_squared_by_channel = host["physical_squared_by_channel"]
    normalized_abs_by_horizon = host["normalized_abs_by_horizon"]
    normalized_squared_by_horizon = host["normalized_squared_by_horizon"]
    physical_abs_by_horizon = host["physical_abs_by_horizon"]
    physical_squared_by_horizon = host["physical_squared_by_horizon"]
    force_absolute_error = float(normalized_abs_by_channel.sum())
    force_squared_error = float(normalized_squared_by_channel.sum())
    confusion_logits = [torch.cat(confusion_logits).cpu()] if confusion_logits else []
    confusion_targets = [torch.cat(confusion_targets).cpu()] if confusion_targets else []
    if collect_predictions:
        force_predictions = [torch.cat(force_predictions).cpu()]
        force_targets = [torch.cat(force_targets).cpu()]
        last_input_forces = [torch.cat(last_input_forces).cpu()]
        shown_labels = [torch.cat(shown_labels).cpu()]
    if not count:
        raise ValueError("the selected data split produced no evaluation batches")
    result: dict[str, Any] = {
        key: value / _dataset_size(loader) for key, value in totals_cpu.items()
    }
    result["classification_accuracy"] = correct_count / count
    result["force_mse_normalized"] = force_squared_error / force_elements
    result["force_mae_normalized"] = force_absolute_error / force_elements
    result["force_rmse_normalized"] = float(np.sqrt(result["force_mse_normalized"]))
    def summarize_errors(absolute: float, squared: float, elements: int) -> dict[str, float]:
        mse = squared / max(elements, 1)
        return {
            "mae": absolute / max(elements, 1),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
        }

    physical_abs = float(physical_abs_by_channel.sum())
    physical_squared = float(physical_squared_by_channel.sum())
    result["regression"] = {
        "units": "original force units",
        "overall": summarize_errors(physical_abs, physical_squared, force_elements),
        "by_channel": {
            channel: summarize_errors(
                physical_abs_by_channel[index],
                physical_squared_by_channel[index],
                channel_elements,
            )
            for index, channel in enumerate(("Fx", "Fy", "Fz"))
        },
        "by_channel_normalized": {
            channel: summarize_errors(
                normalized_abs_by_channel[index],
                normalized_squared_by_channel[index],
                channel_elements,
            )
            for index, channel in enumerate(("Fx", "Fy", "Fz"))
        },
        "by_horizon": [
            summarize_errors(
                physical_abs_by_horizon[index],
                physical_squared_by_horizon[index],
                horizon_elements,
            )
            for index in range(model.config.horizon)
        ],
        "by_horizon_normalized": [
            summarize_errors(
                normalized_abs_by_horizon[index],
                normalized_squared_by_horizon[index],
                horizon_elements,
            )
            for index in range(model.config.horizon)
        ],
        "normalized": summarize_errors(
            force_absolute_error, force_squared_error, force_elements
        ),
    }
    probabilities = softmax_probabilities(torch.cat(confusion_logits))
    targets = torch.cat(confusion_targets).numpy()
    classification, _ = classification_report(probabilities, targets)
    result["classification"] = classification

    normalized_force_targets = torch.cat(force_targets) if collect_predictions else None
    normalized_force_predictions = (
        torch.cat(force_predictions) if collect_predictions else None
    )
    if collect_predictions:
        if normalized_force_targets is None or normalized_force_predictions is None:
            raise RuntimeError("test predictions were not collected")
        mean = normalizer.mean.astype(np.float32)
        scale = normalizer.scale.astype(np.float32)
        prediction_array = normalized_force_predictions.numpy()
        target_array = normalized_force_targets.numpy()
        result["regression"] = regression_report(
            prediction_array, target_array, mean, scale
        )
        probability_array = probabilities
        result["paper_horizons"] = cumulative_horizon_report(
            prediction_array, target_array, probability_array, targets
        )
        shown_label_array = torch.cat(shown_labels).numpy()
        result["baselines"] = baseline_report(
            torch.cat(last_input_forces).numpy(),
            target_array,
            shown_label_array,
            targets,
        )
        result["photo_change"] = photo_change_report(
            probabilities.argmax(axis=-1), targets, shown_label_array
        )
        prediction_array = prediction_array * scale[None, None, :] + mean[None, None, :]
        target_array = target_array * scale[None, None, :] + mean[None, None, :]
        prediction_data = {
            "force_predictions": prediction_array.astype(np.float32),
            "force_targets": target_array.astype(np.float32),
            "class_probabilities": probabilities.astype(np.float32),
            "class_targets": targets.astype(np.int64),
        }
        return result, prediction_data
    return result, None


def train_fold(
    *,
    data_dir: str | Path,
    test_tool: str,
    output_dir: str | Path,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    input_len: int = 96,
    horizon: int = 96,
    stride: int | None = None,
    decimation: int = 1,
    label_column: str = "image_label",
    dedup_identical_forces: bool = True,
    eval_dedup_identical_forces: bool | None = None,
    trim_pass: bool = True,
    image_encoder: str = "efficientnet_b0",
    d_model: int = 32,
    prediction_head: str = "direct",
    max_train_windows: int | None = None,
    max_eval_windows: int | None = None,
    device_name: str = "cpu",
    seed: int = 42,
    pool: int = 1,
    image_lag: int = 0,
    select_best: bool = True,
    accumulation_steps: int = 1,
    num_workers: int = 0,
    resume: bool = True,
    config_overrides: dict[str, Any] | None = None,
    tracker: ExperimentTracker | None = None,
    spectrogram: SpectrogramSettings | None = None,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or accumulation_steps < 1:
        raise ValueError("epochs, batch_size and accumulation_steps must be positive")
    if spectrogram is not None and image_lag != 0:
        raise ValueError(
            "spectrogram images need --image-lag 0: each image is built from the raw force "
            "of the window's own input steps, so it never covers the forecast window and "
            "needs no lag"
        )
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        properties = torch.cuda.get_device_properties(torch.device(device_name))
        if properties.total_memory < 3 * 1024**3:
            raise RuntimeError(
                f"{properties.name} has {properties.total_memory // 1024**2} MiB; "
                "use CPU or a GPU with at least 3 GiB for this model"
            )
    device = torch.device(device_name)
    if device.type == "cpu":
        # Leave the CPUs other programs are using alone (omnifuser.resources).
        torch.set_num_threads(cpu_threads(limit=4))
    if num_workers > available_cpus():
        print(
            f"warning: --num-workers {num_workers} exceeds the {available_cpus()} CPUs "
            "available; data loading will compete with training",
            flush=True,
        )

    # Training and evaluation may handle duplicate recordings differently, so a
    # comparison of the two settings can be scored on identical test windows.
    if eval_dedup_identical_forces is None:
        eval_dedup_identical_forces = dedup_identical_forces

    load = partial(
        load_timelines, data_dir, label_column=label_column, decimation=decimation,
        trim_pass=trim_pass, pool=pool, image_lag=image_lag, spectrogram=spectrogram,
    )
    timelines, retained_records, raw_passes = load(dedup_identical_forces=dedup_identical_forces)
    if eval_dedup_identical_forces == dedup_identical_forces:
        eval_timelines, evaluation_retained_records = timelines, retained_records
        eval_raw_passes = raw_passes
    else:
        eval_timelines, evaluation_retained_records, eval_raw_passes = load(
            dedup_identical_forces=eval_dedup_identical_forces
        )
    source_signature = nonastreda_source_signature(data_dir)
    train_stride = stride if stride is not None else input_len + horizon
    eval_stride = max(train_stride, horizon)
    folds = {fold.test_tools[0]: fold for fold in leave_one_tool_out_folds([t.tool for t in timelines])}
    if test_tool not in folds:
        raise ValueError(f"unknown test tool {test_tool!r}; choose one of {sorted(folds)}")
    fold = folds[test_tool]
    timeline_by_tool = {timeline.tool: timeline for timeline in timelines}
    eval_timeline_by_tool = {timeline.tool: timeline for timeline in eval_timelines}
    train_timelines = [timeline_by_tool[tool] for tool in fold.train_tools]
    validation_timelines = [eval_timeline_by_tool[tool] for tool in fold.validation_tools]
    test_timelines = [eval_timeline_by_tool[tool] for tool in fold.test_tools]
    normalizer = ForceNormalizer.fit(train_timelines)
    config = model_config(
        d_model=d_model,
        input_len=input_len,
        horizon=horizon,
        image_encoder=image_encoder,
        prediction_head=prediction_head,
        config_overrides=config_overrides,
        spectrogram=spectrogram,
    )

    def windows(
        split: list[ToolTimeline], split_stride: int, passes: dict[str, Any]
    ) -> WindowDataset:
        return window_dataset(
            split, normalizer, config, split_stride,
            spectrogram=spectrogram, raw_passes=passes, pool=pool,
        )

    train_data = _limit_dataset(windows(train_timelines, train_stride, raw_passes), max_train_windows, seed)
    validation_data = _limit_dataset(
        windows(validation_timelines, eval_stride, eval_raw_passes), max_eval_windows, seed + 1
    )
    test_data = _limit_dataset(windows(test_timelines, eval_stride, eval_raw_passes), max_eval_windows, seed + 2)

    model = OmniFuser(config).to(device)
    image_bank = image_bank_for([*timelines, *eval_timelines], config, model, device, spectrogram)
    collate = partial(
        collate_windows,
        height=config.image_height,
        width=config.image_width,
        bank=image_bank,
        crop=config.image_crop,
    )
    loader_options: dict[str, Any] = {
        "num_workers": num_workers,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, **loader_options
    )
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False, **loader_options
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, **loader_options
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = f"fold-{test_tool}"
    resume_path = output_path / f"{stem}-resume.pt"
    config_dict = asdict(config)
    resume_signature = {
        "source": source_signature,
        "fold": {
            "train_tools": list(fold.train_tools),
            "validation_tools": list(fold.validation_tools),
            "test_tools": list(fold.test_tools),
        },
        "preprocessing": {
            "label_column": label_column,
            "decimation": decimation,
            "pool": pool,
            "image_lag": image_lag,
            "dedup_identical_forces": dedup_identical_forces,
            "eval_dedup_identical_forces": eval_dedup_identical_forces,
            "trim_pass": trim_pass,
            # Recorded only when used, so photo runs keep their earlier signature.
            **({"spectrogram": asdict(spectrogram)} if spectrogram is not None else {}),
        },
        "windowing": {
            "input_len": input_len,
            "horizon": horizon,
            "stride": train_stride,
            "evaluation_stride": eval_stride,
            "max_train_windows": max_train_windows,
            "max_eval_windows": max_eval_windows,
        },
        "training": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "accumulation_steps": accumulation_steps,
            "num_workers": num_workers,
            "seed": seed,
            "device": str(device),
            "select_best": select_best,
        },
        "config": config_dict,
    }
    if tracker is None:
        gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
        tracker = ExperimentTracker(
            output_path,
            {
                "dataset": "Nonastreda",
                "data_dir": str(Path(data_dir).resolve()),
                "source_signature": source_signature,
                "fold": resume_signature["fold"],
                "preprocessing": resume_signature["preprocessing"],
                "windowing": resume_signature["windowing"],
                "training": {
                    **resume_signature["training"],
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                },
                "model": config_dict,
                "evaluation_modes": ["full", "sensor_missing", "image_missing"],
                "runtime": {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "cuda_runtime": torch.version.cuda,
                    "device": str(device),
                    "gpu_name": gpu_name,
                    "gpu_total_memory_bytes": (
                        torch.cuda.get_device_properties(device).total_memory
                        if device.type == "cuda"
                        else None
                    ),
                },
            },
        )
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("inf")
    best_epoch = 0
    start_epoch = 1
    saved: dict[str, Any] | None = None
    if resume and resume_path.exists():
        try:
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        except (EOFError, OSError, RuntimeError, pickle.UnpicklingError):
            print(f"{stem}: ignoring an unreadable resume file; restarting this fold", flush=True)
    if saved is not None:
        if (
            saved.get("signature") == resume_signature
            and saved.get("epoch", epochs + 1) <= epochs
            and "rng_state" in saved
        ):
            model.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            history = saved["history"]
            best_state = saved["best_state"]
            best_score = saved["best_score"]
            best_epoch = saved["best_epoch"]
            start_epoch = saved["epoch"] + 1
            _restore_rng_state(saved["rng_state"])
            print(f"{stem}: resuming after epoch {saved['epoch']}", flush=True)
        else:
            print(
                f"{stem}: ignoring a resume file written with other or incomplete settings",
                flush=True,
            )
    tracker.log(
        split="setup",
        fold=test_tool,
        metrics={
            "normalizer_mean": normalizer.mean.tolist(),
            "normalizer_scale": normalizer.scale.tolist(),
            "retained_records": retained_records,
            "evaluation_retained_records": evaluation_retained_records,
            "window_counts": {
                "train": len(train_data),
                "validation": len(validation_data),
                "test": len(test_data),
            },
        },
        context={
            "train_tools": list(fold.train_tools),
            "validation_tool": fold.validation_tools[0],
            "test_tool": test_tool,
            "source_signature": source_signature,
        },
    )
    for epoch in range(start_epoch, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_started = time.perf_counter()
        train_metrics, _ = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            normalizer,
            accumulation_steps=accumulation_steps,
        )
        train_seconds = time.perf_counter() - train_started
        train_memory = _cuda_memory_metrics(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        validation_started = time.perf_counter()
        validation_metrics, _ = _run_epoch(
            model, validation_loader, None, device, normalizer
        )
        validation_seconds = time.perf_counter() - validation_started
        tracker_context = {
            "train_windows": len(train_data),
            "validation_windows": len(validation_data),
            "learning_rate": learning_rate,
        }
        tracker.log(
            split="train",
            fold=test_tool,
            epoch=epoch,
            metrics={
                **train_metrics,
                "epoch_seconds": train_seconds,
                **{f"cuda_{key}": value for key, value in train_memory.items()},
            },
            context=tracker_context,
        )
        tracker.log(
            split="validation",
            fold=test_tool,
            epoch=epoch,
            metrics={
                **validation_metrics,
                "epoch_seconds": validation_seconds,
                **{
                    f"cuda_{key}": value
                    for key, value in _cuda_memory_metrics(device).items()
                },
            },
            context=tracker_context,
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        # One progress line per epoch, so a log shows a long fold is advancing.
        print(
            f"  {test_tool} epoch {epoch}/{epochs}: "
            f"train loss {train_metrics['loss']:.4f}, "
            f"validation loss {validation_metrics['loss']:.4f}, "
            f"{train_seconds + validation_seconds:.0f} s",
            flush=True,
        )
        if validation_metrics["loss"] < best_score:
            best_score = validation_metrics["loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        _atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "best_state": best_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "epoch": epoch,
                "signature": resume_signature,
                "rng_state": _capture_rng_state(),
            },
            resume_path,
        )

    if select_best and best_state is not None:
        model.load_state_dict(best_state)
    selected_epoch = best_epoch if select_best else epochs

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    test_started = time.perf_counter()
    test_metrics, test_predictions = _run_epoch(
        model,
        test_loader,
        None,
        device,
        normalizer,
        collect_predictions=True,
    )
    if test_predictions is None:
        raise RuntimeError("test evaluation did not return predictions")
    test_metrics["baselines"]["run_number_class_accuracy"] = run_number_baseline_accuracy(
        train_timelines, test_data
    )
    test_seconds = time.perf_counter() - test_started
    tracker.log(
        split="test",
        fold=test_tool,
        epoch=selected_epoch,
        metrics={
            **test_metrics,
            "evaluation_seconds": test_seconds,
            **{
                f"cuda_{key}": value
                for key, value in _cuda_memory_metrics(device).items()
            },
        },
        context={
            "test_windows": len(test_data),
            "selected_epoch": selected_epoch,
            "model_selection": "best_validation_loss" if select_best else "last_epoch",
            "evaluation_mode": "full",
            "selected_checkpoint_evaluation": True,
        },
    )
    modality_evaluations: dict[str, dict[str, Any]] = {
        "full": {"train": {}, "validation": {}, "test": test_metrics},
        "sensor_missing": {},
        "image_missing": {},
    }
    for stage, loader in (
        ("train", train_loader),
        ("validation", validation_loader),
    ):
        for mode, dropped in (
            ("full", None),
            ("sensor_missing", "sensor"),
            ("image_missing", "image"),
        ):
            modality_metrics, _ = _run_epoch(
                model,
                loader,
                None,
                device,
                normalizer,
                drop_modality=dropped,
            )
            modality_evaluations[mode][stage] = modality_metrics
            tracker.log(
                split=stage,
                fold=test_tool,
                epoch=selected_epoch,
                metrics=modality_metrics,
                context={
                    "evaluation_mode": mode,
                    "selected_checkpoint_evaluation": True,
                    "windows": _dataset_size(loader),
                },
            )

    modality_artifacts: dict[str, str] = {}
    for mode, dropped in (
        ("sensor_missing", "sensor"),
        ("image_missing", "image"),
    ):
        modality_metrics, modality_predictions = _run_epoch(
            model,
            test_loader,
            None,
            device,
            normalizer,
            collect_predictions=True,
            drop_modality=dropped,
        )
        if modality_predictions is None:
            raise RuntimeError(f"{mode} test evaluation did not return predictions")
        modality_evaluations[mode]["test"] = modality_metrics
        tracker.log(
            split="test",
            fold=test_tool,
            epoch=selected_epoch,
            metrics=modality_metrics,
            context={
                "evaluation_mode": mode,
                "selected_checkpoint_evaluation": True,
                "test_windows": len(test_data),
            },
        )
        _, mode_roc_curves = classification_report(
            modality_predictions["class_probabilities"],
            modality_predictions["class_targets"],
        )
        prediction_name = f"{stem}-predictions-{mode}.npz"
        roc_name = f"{stem}-roc-{mode}.npz"
        save_arrays(output_path / prediction_name, modality_predictions)
        save_roc_curves(output_path / roc_name, mode_roc_curves)
        modality_artifacts[f"{mode}_predictions"] = prediction_name
        modality_artifacts[f"{mode}_roc_curves"] = roc_name

    _, roc_curves = classification_report(
        test_predictions["class_probabilities"], test_predictions["class_targets"]
    )
    result: dict[str, Any] = {
        "test_tool": test_tool,
        "validation_tool": fold.validation_tools[0],
        "train_tools": list(fold.train_tools),
        "prediction_head": config.prediction_head,
        "label_column": label_column,
        "dedup_identical_forces": dedup_identical_forces,
        "eval_dedup_identical_forces": eval_dedup_identical_forces,
        "retained_records": retained_records,
        "evaluation_retained_records": evaluation_retained_records,
        "trim_pass": trim_pass,
        "pass_trimming": "span" if trim_pass else "none",
        "pass_trimming_parameters": (
            {
                "envelope_window_samples": PASS_TRIM_WINDOW_SAMPLES,
                "early_reference_fraction": PASS_TRIM_REFERENCE_FRACTION,
                "active_threshold_fraction": PASS_TRIM_THRESHOLD_FRACTION,
            }
            if trim_pass
            else None
        ),
        "decimation": decimation,
        "pool": pool,
        "image_lag": image_lag,
        **({"spectrogram": asdict(spectrogram)} if spectrogram is not None else {}),
        "stride": train_stride,
        "evaluation_stride": eval_stride,
        "model_selection": (
            "best_validation_loss" if select_best else "last_epoch"
        ),
        "selected_epoch": selected_epoch,
        "best_validation_epoch": best_epoch,
        "source_signature": source_signature,
        "max_train_windows": max_train_windows,
        "max_eval_windows": max_eval_windows,
        "training": {
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch_size": batch_size * accumulation_steps,
            "epochs": epochs,
            "seed": seed,
            "device": str(device),
            "num_workers": num_workers,
            "cached_image_features": isinstance(image_bank, FrozenFeatureBank),
        },
        "normalizer_mean": normalizer.mean.tolist(),
        "normalizer_scale": normalizer.scale.tolist(),
        "window_counts": {
            "train": len(train_data),
            "validation": len(validation_data),
            "test": len(test_data),
        },
        "config": asdict(config),
        "effective_architecture": {
            "temporal_extractor": (
                "mrte"
                if hasattr(model.temporal_encoder, "layers")
                else "identity"
            ),
            "rtd_layers": len(getattr(model.temporal_encoder, "layers", ())),
            "image_gat_layers": len(model.image_branch.layers),
            "c2f_rounds": len(model.fusion_layers),
        },
        "history": history,
        "test": test_metrics,
        "modality_evaluations": modality_evaluations,
        "tracking": {"directory": "tracking", "run_id": tracker.run_id},
        "artifacts": {
            "checkpoint": f"fold-{test_tool}.pt",
            "test_predictions": f"fold-{test_tool}-predictions.npz",
            "roc_curves": f"fold-{test_tool}-roc.npz",
            **modality_artifacts,
        },
    }
    save_arrays(output_path / f"{stem}-predictions.npz", test_predictions)
    save_roc_curves(output_path / f"{stem}-roc.npz", roc_curves)
    # The fold's final model (the selected epoch's weights) as a portable model file:
    # reusable without training via model_bundle.load_model / 'omnifuser predict'.
    _atomic_torch_save(
        make_bundle(asdict(config), model.state_dict(), normalizer.mean, normalizer.scale, result),
        output_path / f"{stem}.pt",
    )
    # The fold JSON is the completion marker that fold reuse looks for, so it is
    # written last: an interrupted fold never looks finished.
    json_path = output_path / f"{stem}.json"
    with _temporary_path(json_path).open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    os.replace(_temporary_path(json_path), json_path)
    resume_path.unlink(missing_ok=True)
    return result
