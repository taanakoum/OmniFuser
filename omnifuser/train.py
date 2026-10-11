from __future__ import annotations

import json
import random
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from omnifuser.config import ModelConfig
from omnifuser.data import (
    ForceNormalizer,
    ImageBank,
    PASS_TRIM_REFERENCE_FRACTION,
    PASS_TRIM_THRESHOLD_FRACTION,
    PASS_TRIM_WINDOW_SAMPLES,
    WindowDataset,
    collate_windows,
    leave_one_tool_out_folds,
    load_nonastreda,
    make_tool_timelines,
    nonastreda_source_signature,
)
from omnifuser.evaluation import (
    baseline_report,
    classification_report,
    cumulative_horizon_report,
    regression_report,
    softmax_probabilities,
)
from omnifuser.models import OmniFuser, multitask_loss


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


def _run_epoch(
    model: OmniFuser,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    normalizer: ForceNormalizer,
    *,
    collect_predictions: bool = False,
    accumulation_steps: int = 1,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    correct = 0
    count = 0
    force_squared_error = 0.0
    force_absolute_error = 0.0
    force_elements = 0
    confusion_logits: list[torch.Tensor] = []
    confusion_targets: list[torch.Tensor] = []
    force_predictions: list[torch.Tensor] = []
    force_targets: list[torch.Tensor] = []
    last_input_forces: list[torch.Tensor] = []
    shown_labels: list[torch.Tensor] = []
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        batches = len(loader)
        if training:
            optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            tensors = {
                key: value.to(device)
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
            )
            loss, components = multitask_loss(
                outputs, tensors["force_target"], tensors["class_target"]
            )
            if training:
                if loader.batch_size is None:
                    raise ValueError("gradient accumulation requires a fixed DataLoader batch size")
                group_samples = _accumulation_group_sample_count(
                    len(loader.dataset), loader.batch_size, step, accumulation_steps
                )
                batch_samples = tensors["force"].shape[0]
                (loss * (batch_samples / group_samples)).backward()
                if step % accumulation_steps == 0 or step == batches:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            batch_size = tensors["force"].shape[0]
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size
            predicted = outputs["class_logits"].argmax(dim=-1)
            correct += int((predicted == tensors["class_target"]).sum())
            count += tensors["class_target"].numel()
            confusion_logits.append(outputs["class_logits"].detach().cpu())
            confusion_targets.append(tensors["class_target"].detach().cpu())
            if collect_predictions:
                force_predictions.append(outputs["force"].detach().cpu())
                force_targets.append(tensors["force_target"].detach().cpu())
                last_input_forces.append(tensors["force"][:, -1].detach().cpu())
                shown_labels.append(tensors["shown_label"].detach().cpu())
            difference = outputs["force"] - tensors["force_target"]
            force_squared_error += float(difference.detach().square().sum())
            force_absolute_error += float(difference.detach().abs().sum())
            force_elements += difference.numel()
    if not count:
        raise ValueError("the selected data split produced no evaluation batches")
    result: dict[str, Any] = {
        key: value / len(loader.dataset) for key, value in totals.items()
    }
    result["classification_accuracy"] = correct / count
    result["force_mse_normalized"] = force_squared_error / force_elements
    result["force_mae_normalized"] = force_absolute_error / force_elements
    result["force_rmse_normalized"] = float(np.sqrt(result["force_mse_normalized"]))
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
        result["baselines"] = baseline_report(
            torch.cat(last_input_forces).numpy(),
            target_array,
            torch.cat(shown_labels).numpy(),
            targets,
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
    horizon: int = 24,
    stride: int | None = None,
    decimation: int = 1,
    label_column: str = "image_label",
    dedup_identical_forces: bool = True,
    trim_pass: bool = True,
    image_encoder: str = "efficientnet_b0",
    d_model: int = 32,
    prediction_head: str = "lead_conditioned",
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
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or accumulation_steps < 1:
        raise ValueError("epochs, batch_size and accumulation_steps must be positive")
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
        torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    records = load_nonastreda(
        data_dir,
        label_column=label_column,
        decimation=decimation,
        dedup_identical_forces=dedup_identical_forces,
        trim_pass=trim_pass,
        pool=pool,
    )
    timelines = make_tool_timelines(records, image_lag=image_lag)
    del records
    source_signature = nonastreda_source_signature(data_dir)
    train_stride = stride if stride is not None else input_len + horizon
    eval_stride = max(train_stride, horizon)
    folds = {fold.test_tools[0]: fold for fold in leave_one_tool_out_folds([t.tool for t in timelines])}
    if test_tool not in folds:
        raise ValueError(f"unknown test tool {test_tool!r}; choose one of {sorted(folds)}")
    fold = folds[test_tool]
    timeline_by_tool = {timeline.tool: timeline for timeline in timelines}
    train_timelines = [timeline_by_tool[tool] for tool in fold.train_tools]
    validation_timelines = [timeline_by_tool[tool] for tool in fold.validation_tools]
    test_timelines = [timeline_by_tool[tool] for tool in fold.test_tools]
    normalizer = ForceNormalizer.fit(train_timelines)
    train_data = WindowDataset(train_timelines, normalizer, input_len, horizon, train_stride)
    validation_data = WindowDataset(
        validation_timelines, normalizer, input_len, horizon, eval_stride
    )
    test_data = WindowDataset(test_timelines, normalizer, input_len, horizon, eval_stride)
    train_data = _limit_dataset(train_data, max_train_windows, seed)
    validation_data = _limit_dataset(validation_data, max_eval_windows, seed + 1)
    test_data = _limit_dataset(test_data, max_eval_windows, seed + 2)

    config = ModelConfig(
        d_model=d_model,
        input_len=input_len,
        horizon=horizon,
        image_encoder=image_encoder,
        prediction_head=prediction_head,
        **(config_overrides or {}),
    )
    image_bank = ImageBank.from_timelines(
        timelines, config.image_height, config.image_width
    )
    collate = partial(
        collate_windows,
        height=config.image_height,
        width=config.image_width,
        bank=image_bank,
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
    model = OmniFuser(config).to(device)
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
            "trim_pass": trim_pass,
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
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("inf")
    best_epoch = 0
    start_epoch = 1
    if resume and resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
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
    for epoch in range(start_epoch, epochs + 1):
        train_metrics, _ = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            normalizer,
            accumulation_steps=accumulation_steps,
        )
        validation_metrics, _ = _run_epoch(
            model, validation_loader, None, device, normalizer
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        if validation_metrics["loss"] < best_score:
            best_score = validation_metrics["loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        torch.save(
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
        },
        "normalizer_mean": normalizer.mean.tolist(),
        "normalizer_scale": normalizer.scale.tolist(),
        "window_counts": {
            "train": len(train_data),
            "validation": len(validation_data),
            "test": len(test_data),
        },
        "config": asdict(config),
        "history": history,
        "test": test_metrics,
        "artifacts": {
            "checkpoint": f"fold-{test_tool}.pt",
            "test_predictions": f"fold-{test_tool}-predictions.npz",
            "roc_curves": f"fold-{test_tool}-roc.npz",
        },
    }
    with (output_path / f"{stem}.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    np.savez_compressed(output_path / f"{stem}-predictions.npz", **test_predictions)
    np.savez_compressed(
        output_path / f"{stem}-roc.npz",
        **{
            f"{name}_{coordinate}": values
            for name, curve in roc_curves.items()
            for coordinate, values in zip(("fpr", "tpr", "thresholds"), curve)
        },
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "normalizer_mean": normalizer.mean,
            "normalizer_scale": normalizer.scale,
            "result": result,
        },
        output_path / f"{stem}.pt",
    )
    resume_path.unlink(missing_ok=True)
    return result
