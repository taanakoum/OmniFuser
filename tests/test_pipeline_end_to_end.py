"""End-to-end 10-fold run on a synthetic Nonastreda-shaped dataset."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest import mock

import numpy as np
from test_alignment_patch import make_nonastreda

from omnifuser import cross_validation
from omnifuser.cross_validation import EVALUATION_MODES, run_cross_validation
from omnifuser.data import ForceNormalizer, load_nonastreda, make_tool_timelines


def _run(root: Path, output: Path) -> dict:
    return run_cross_validation(
        data_dir=root,
        output_dir=output,
        epochs=1,
        batch_size=2,
        input_len=24,
        horizon=12,
        stride=6,
        image_encoder="compact",
        d_model=16,
        prediction_head="direct",
        device_name="cpu",
        max_train_windows=4,
        max_eval_windows=2,
        pool=2,
        image_lag=1,
        trim_pass=False,
        config_overrides={
            "num_rtd_layers": 2,
            "num_gat_layers": 1,
            "image_height": 32,
            "image_width": 96,
            "modality_dropout": 0.2,
            "ablation": "without_hybrid_gate",
        },
    )


def test_cross_validation_tracks_all_modes_and_reuses_completed_folds(tmp_path: Path) -> None:
    root = make_nonastreda(tmp_path / "data", tools=tuple(range(1, 11)), runs=(1, 2), length=200)
    output = tmp_path / "cv"

    summary = _run(root, output)

    assert summary["cv_mean_std"]["folds"] == 10
    assert summary["ablation"] == "without_hybrid_gate"
    assert summary["config"]["modality_dropout"] == 0.2
    for mode in EVALUATION_MODES:
        for stage in ("train", "validation", "test"):
            metrics = summary["modality_evaluations"][mode][stage]
            for name in ("classification_accuracy", "classification_f1_macro",
                         "regression_mse", "regression_mse_normalized"):
                assert len(metrics[name]["per_fold"]) == 10
                assert None not in metrics[name]["per_fold"], (mode, stage, name)
    # the full-input test summary is the same evaluation as the headline CV metrics
    assert np.isclose(
        summary["modality_evaluations"]["full"]["test"]["classification_accuracy"]["mean"],
        summary["cv_mean_std"]["metrics"]["classification_accuracy"]["mean"],
    )
    assert summary["out_of_fold"]["evaluation_mode"] == "full"
    assert "normalized" in summary["out_of_fold"]["regression"]
    for mode in ("sensor_missing", "image_missing"):
        pooled = summary["out_of_fold_by_mode"][mode]
        assert pooled["evaluation_mode"] == mode
        assert (output / pooled["classification"]["mean_roc_curve_file"]).is_file()
        assert sum(map(sum, pooled["classification"]["confusion_matrix"])) == sum(
            map(sum, summary["out_of_fold"]["classification"]["confusion_matrix"])
        )

    records = load_nonastreda(root, trim_pass=False, pool=2)
    timelines = make_tool_timelines(records, image_lag=1)
    for fold_file in sorted(output.glob("fold-T*.json")):
        result = json.loads(fold_file.read_text(encoding="utf-8"))
        assert result["test_tool"] not in result["train_tools"]
        assert result["validation_tool"] not in result["train_tools"]
        assert result["stride"] == 6 and result["evaluation_stride"] == 12
        assert result["effective_architecture"]["c2f_rounds"] == 2
        for artifact in result["artifacts"].values():
            assert (output / artifact).is_file()
        # normalization statistics come from the eight training tools only
        expected = ForceNormalizer.fit(
            [timeline for timeline in timelines if timeline.tool in result["train_tools"]]
        )
        assert np.allclose(result["normalizer_mean"], expected.mean, rtol=1e-5)
        assert np.allclose(result["normalizer_scale"], expected.scale, rtol=1e-5)
        with np.load(output / result["artifacts"]["sensor_missing_predictions"]) as data:
            assert data["force_predictions"].shape[1:] == (12, 3)
            assert data["class_probabilities"].shape[1:] == (12, 3)

    with (output / "tracking" / "metrics.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    logged = {(row["split"], row["evaluation_mode"]) for row in rows}
    for split in ("train", "validation", "test"):
        for mode in EVALUATION_MODES:
            assert (split, mode) in logged
    epoch_metrics = {row["metric"] for row in rows if row["split"] == "train" and row["evaluation_mode"] == ""}
    assert {"loss", "classification.f1_macro", "classification.precision_weighted",
            "regression.overall.mae", "regression.normalized.mse"} <= epoch_metrics
    assert len({row["fold"] for row in rows if row["split"] == "test"}) == 10

    # A rerun with identical settings reuses every completed fold without training.
    with mock.patch.object(cross_validation, "train_fold", side_effect=AssertionError("retrained")):
        reused = _run(root, output)
    assert reused["cv_mean_std"] == summary["cv_mean_std"]
    assert reused["modality_evaluations"] == summary["modality_evaluations"]
