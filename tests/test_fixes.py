"""Training-side classification fixes: metadata switch, RevIN, ordinal head, photo crop,
and the run-number and photo-change reference metrics."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from omnifuser.cli import build_parser, protocol_kwargs
from omnifuser.config import ModelConfig
from omnifuser.data import (
    ForceNormalizer,
    ImageBank,
    ToolTimeline,
    WindowDataset,
    image_tensor,
)
from omnifuser.evaluation import photo_change_report, run_number_baseline_accuracy
from omnifuser.models import OmniFuser, multitask_loss, ordinal_log_probabilities


def _config(**changes: object) -> ModelConfig:
    base = ModelConfig(
        d_model=16, input_len=12, horizon=4, num_rtd_layers=2, rtd_rank=4,
        num_gat_layers=1, num_gat_heads=4, num_fusion_layers=2, num_landmarks=2,
        image_encoder="compact", image_height=32, image_width=64, dropout=0.0,
    )
    return replace(base, **changes)


def _inputs(batch: int = 2, steps: int = 12) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "force": torch.randn(batch, steps, 3, generator=generator),
        "images": torch.randn(batch, 2, 3, 32, 64, generator=generator),
        "image_indices": torch.zeros(batch, steps, dtype=torch.long),
        "cycle": torch.full((batch, steps), 3),
        "batch_code": torch.full((batch, steps), 1),
        "relative": torch.arange(steps).repeat(batch, 1) % 16,
    }


def _run(model: OmniFuser, **overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    inputs = {**_inputs(), **overrides}
    return model(inputs["force"], inputs["images"], inputs["image_indices"],
                 inputs["cycle"], inputs["batch_code"], inputs["relative"])


@pytest.mark.parametrize(("metadata", "uses_run", "uses_position"),
                         [("all", True, True), ("position", False, True), ("none", False, False)])
def test_metadata_switch_removes_the_chosen_codes(metadata: str, uses_run: bool, uses_position: bool) -> None:
    torch.manual_seed(0)
    model = OmniFuser(_config(metadata=metadata)).eval()
    keys = model.state_dict().keys()
    assert ("temporal_embedding.cycle.weight" in keys) == uses_run
    assert ("image_branch.batch.weight" in keys) == uses_run
    assert ("temporal_embedding.relative.weight" in keys) == uses_position
    reference = _run(model)["class_logits"]
    other_run = _run(model, cycle=torch.full((2, 12), 9), batch_code=torch.full((2, 12), 4))["class_logits"]
    other_position = _run(model, relative=torch.zeros(2, 12, dtype=torch.long))["class_logits"]
    assert torch.allclose(reference, other_run) != uses_run
    assert torch.allclose(reference, other_position) != uses_position


def test_revin_forecast_follows_a_shift_of_the_input_level() -> None:
    torch.manual_seed(0)
    model = OmniFuser(_config(revin=True)).eval()
    inputs = _inputs()
    shift = torch.tensor([1.5, -2.0, 0.7])
    with torch.no_grad():
        base = _run(model)
        moved = _run(model, force=inputs["force"] + shift)
    assert torch.allclose(moved["force"], base["force"] + shift, atol=1e-5)
    assert torch.allclose(moved["class_logits"], base["class_logits"], atol=1e-5)
    plain = OmniFuser(_config()).eval()
    with torch.no_grad():
        assert not torch.allclose(_run(plain, force=inputs["force"] + shift)["force"], _run(plain)["force"] + shift)


def test_ordinal_log_probabilities_are_differences_of_exceedances() -> None:
    cumulative = torch.tensor([[2.0, -1.0]])
    expected = torch.tensor([[1 - torch.sigmoid(torch.tensor(2.0)),
                              torch.sigmoid(torch.tensor(2.0)) - torch.sigmoid(torch.tensor(-1.0)),
                              torch.sigmoid(torch.tensor(-1.0))]])
    assert torch.allclose(ordinal_log_probabilities(cumulative).exp(), expected)


def test_coral_head_gives_valid_ordered_probabilities_and_its_loss() -> None:
    torch.manual_seed(0)
    model = OmniFuser(_config(class_loss="coral"))
    outputs = _run(model)
    cumulative = outputs["class_cumulative_logits"]
    assert cumulative.shape == (2, 4, 2)
    assert (cumulative[..., 0] >= cumulative[..., 1]).all()  # P(> sharp) >= P(> used)
    assert torch.allclose(outputs["class_logits"].exp().sum(-1), torch.ones(2, 4), atol=1e-6)
    assert torch.allclose(torch.softmax(outputs["class_logits"], -1), outputs["class_logits"].exp(), atol=1e-6)
    targets = torch.tensor([[0, 1, 2, 2], [1, 1, 0, 2]])
    loss, components = multitask_loss(outputs, torch.zeros(2, 4, 3), targets)
    above = torch.stack([(targets > 0), (targets > 1)], dim=-1).float()
    manual = F.binary_cross_entropy_with_logits(cumulative, above, reduction="none").sum(-1).mean()
    assert torch.isclose(components["class_loss"], manual)
    loss.backward()
    assert model.coral_gaps.grad is not None and model.direct_class.weight.grad is not None
    with pytest.raises(ValueError, match="direct prediction head"):
        _config(class_loss="coral", prediction_head="lead_conditioned")


def test_image_crop_is_applied_before_resizing(tmp_path: Path) -> None:
    rows = np.zeros((100, 200, 3), dtype=np.uint8)
    rows[50:, :, :] = 255  # bright lower half
    path = tmp_path / "photo.png"
    Image.fromarray(rows).save(path)
    lower = image_tensor(path, 32, 64, crop=(0.5, 0.0, 1.0, 1.0))
    whole = image_tensor(path, 32, 64)
    assert torch.allclose(lower, lower[:, :1, :1].expand_as(lower))  # all bright
    assert not torch.allclose(whole, whole[:, :1, :1].expand_as(whole))
    bank = ImageBank([path], 32, 64, crop=(0.5, 0.0, 1.0, 1.0))
    assert torch.allclose(bank.tensor(path), lower, atol=1e-6)
    with pytest.raises(ValueError, match="image_crop"):
        _config(image_crop=(0.6, 0.0, 0.5, 1.0))


def test_crop_survives_a_json_round_trip_as_saved_folds_and_models_store_it() -> None:
    config = _config(image_crop=(0.3, 0.0, 0.85, 1.0), metadata="position", revin=True)
    restored = ModelConfig(**json.loads(json.dumps(asdict(config))))
    assert restored == config and isinstance(restored.image_crop, tuple)
    # A config written before the new fields existed takes their defaults.
    old = {key: value for key, value in asdict(_config()).items()
           if key not in {"image_crop", "metadata", "revin", "class_loss"}}
    assert asdict(ModelConfig(**old)) == asdict(_config())


def test_cli_passes_the_fix_options_to_the_model_config() -> None:
    args = build_parser().parse_args([
        "train", "--metadata", "position", "--revin", "--class-loss", "coral",
        "--image-size", "256", "1280", "--image-crop", "0.3", "0", "0.85", "1",
    ])
    overrides = protocol_kwargs(args)["config_overrides"]
    assert overrides["metadata"] == "position" and overrides["revin"] and overrides["class_loss"] == "coral"
    assert (overrides["image_height"], overrides["image_width"]) == (256, 1280)
    assert overrides["image_crop"] == (0.3, 0.0, 0.85, 1.0)
    defaults = protocol_kwargs(build_parser().parse_args(["train"]))["config_overrides"]
    assert defaults["metadata"] == "all" and not defaults["revin"] and defaults["image_crop"] is None
    assert "image_height" not in defaults


def test_photo_change_report_splits_accuracy_by_label_change() -> None:
    targets = np.array([[0, 0, 1], [2, 2, 2]])
    predicted = np.array([[0, 1, 1], [2, 2, 1]])
    shown = np.array([0, 2])
    report = photo_change_report(predicted, targets, shown)
    assert report["label_changed_share"] == 1 / 6
    assert report["accuracy_label_changed"] == 1.0
    assert report["accuracy_label_unchanged"] == 3 / 5


def _timeline(tool: str, runs: list[int], labels: list[int], steps_per_run: int = 4) -> ToolTimeline:
    run_codes = np.repeat(runs, steps_per_run).astype(np.int16)
    length = len(run_codes)
    return ToolTimeline(
        tool=tool, force=np.zeros((length, 3), dtype=np.float32),
        labels=np.repeat(labels, steps_per_run).astype(np.int8),
        image_codes=np.zeros(length, dtype=np.int16), cycle_codes=run_codes,
        batch_codes=np.ones(length, dtype=np.int16), relative_codes=np.zeros(length, dtype=np.int16),
        image_paths=(Path("x.jpg"),), image_labels=np.repeat(labels, steps_per_run).astype(np.int8),
    )


def test_run_number_baseline_uses_the_training_tools_majority_class_per_run() -> None:
    train = [_timeline("T1", [1, 2, 3], [0, 1, 2]), _timeline("T2", [1, 2, 3], [0, 1, 1])]
    # run 3 is "dulled" in T1 and "used" in T2: the tie goes to the lower class (argmax)
    test = _timeline("T3", [1, 2, 3, 4], [0, 2, 1, 2])
    normalizer = ForceNormalizer(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
    windows = WindowDataset([test], normalizer, input_len=4, horizon=12, stride=4)
    # one window, targets in runs 2, 3, 4 (classes 2, 1, 2); the rule predicts 1, 1 and,
    # for the unseen run 4, the most frequent training class 1: only run 3 is right
    assert run_number_baseline_accuracy(train, windows) == pytest.approx(4 / 12)


def test_a_fold_trains_and_reloads_with_every_fix_option(tmp_path: Path) -> None:
    from test_alignment_patch import make_nonastreda

    from omnifuser.model_bundle import load_model, predict
    from omnifuser.train import train_fold

    root = make_nonastreda(tmp_path / "data", tools=(1, 2, 3), runs=(1,), length=120)
    overrides = {"num_rtd_layers": 2, "num_gat_layers": 1, "image_height": 32, "image_width": 96,
                 "metadata": "none", "revin": True, "class_loss": "coral", "image_crop": (0.3, 0.0, 0.85, 1.0)}
    result = train_fold(
        data_dir=root, test_tool="T1", output_dir=tmp_path / "run", epochs=1, batch_size=2,
        input_len=24, horizon=12, stride=12, device_name="cpu", trim_pass=False, pool=4,
        image_encoder="compact", max_train_windows=6, num_workers=0, config_overrides=overrides,
    )
    test = result["test"]
    assert 0.0 <= test["baselines"]["run_number_class_accuracy"] <= 1.0
    assert set(test["photo_change"]) == {"label_changed_share", "accuracy_label_changed", "accuracy_label_unchanged"}
    loaded = load_model(tmp_path / "run" / "fold-T1.pt")
    assert loaded.config.image_crop == (0.3, 0.0, 0.85, 1.0) and loaded.config.class_loss == "coral"
    again = predict(tmp_path / "run" / "fold-T1.pt", root, "T1", tmp_path / "predicted")
    assert again["classification_accuracy"] == pytest.approx(test["classification_accuracy"])
    assert again["force_mse_normalized"] == pytest.approx(test["force_mse_normalized"], rel=1e-5)
    denser = predict(tmp_path / "run" / "fold-T1.pt", root, "T1", tmp_path / "predicted", evaluation_stride=4)
    assert denser["prediction"]["windows"] > again["prediction"]["windows"]
    assert (tmp_path / "predicted" / "predictions-T1-stride4.npz").is_file()
