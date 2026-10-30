import csv
import json
from pathlib import Path
from unittest import mock

import pytest

import scripts.run_experiment_suite as suite
from scripts.run_experiment_suite import (
    _ablation_runs,
    _experiment_runs,
    _missing_modality_runs,
    _sensitivity_runs,
    build_parser,
)


def test_paper_ablation_suite_contains_full_model_and_all_reported_variants() -> None:
    runs = _ablation_runs()
    assert len(runs) == 9
    assert {run["ablation"] for run in runs} == {
        "none",
        "without_rr",
        "without_mrte",
        "without_rr_mrte",
        "concat_c2f",
        "without_c2f",
        "without_als",
        "without_hybrid_gate",
        "moving_average",
    }


def _swept(runs: list[dict], field: str) -> set:
    return {run["overrides"][field] for run in runs if field in run["overrides"]}


def test_sensitivity_suite_sweeps_every_paper_axis_without_baseline_duplicates() -> None:
    runs = _sensitivity_runs()

    # Figs. 7-8: k, d, GAT heads, r (RTD trend rank), GAT layers, s_l, L, n.
    assert _swept(runs, "num_landmarks") == {2, 8, 16}
    assert _swept(runs, "d_model") == {16, 64, 128}
    assert _swept(runs, "num_gat_heads") == {2, 8, 16}
    assert _swept(runs, "rtd_rank") == {4, 16, 32}
    assert _swept(runs, "num_gat_layers") == {2, 3}
    assert _swept(runs, "rtd_stride") == {1, 2}
    assert _swept(runs, "num_rtd_layers") == {1, 2, 3}
    assert _swept(runs, "num_fusion_layers") == {1, 3}
    assert not _swept(runs, "fusion_rank")
    assert runs[0]["name"] == "sensitivity-baseline" and runs[0]["overrides"] == {}
    assert len(runs) == 22
    assert len({run["name"] for run in runs}) == len(runs)


def test_sensitivity_skips_values_of_a_changed_baseline() -> None:
    from dataclasses import asdict

    from omnifuser.config import ModelConfig

    baseline = asdict(ModelConfig(num_landmarks=8))
    assert _swept(_sensitivity_runs(baseline), "num_landmarks") == {2, 4, 16}


def test_missing_modality_study_has_reference_and_requested_rates() -> None:
    runs = _missing_modality_runs((0.1, 0.3))
    assert [run["overrides"]["modality_dropout"] for run in runs] == [0.0, 0.1, 0.3]
    assert runs[1]["name"] == "modality-dropout-0-1"
    assert len(_missing_modality_runs((0.2,), include_reference=False)) == 1


def test_experiment_suite_parser_exposes_studies_and_durable_output_layout() -> None:
    args = build_parser().parse_args([])

    assert args.study == "ablations"
    assert str(args.output_root) == "experiment_results"
    assert args.modality_dropout_rates == [0.2]
    # protocol defaults: 1 s steps and the same-pass photo (paper)
    assert (args.pool, args.image_lag, args.stride, args.horizon, args.epochs) == (1000, 0, 24, 96, 30)
    # combined study: 9 ablation arms + 21 sensitivity points + 1 dropout arm
    assert len(_experiment_runs("all")) == 31


def _fake_cross_validation(**kwargs):
    output = Path(kwargs["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    metric = {"mean": kwargs["config_overrides"].get("modality_dropout", 0.0), "std": 0.0}
    evaluations = {
        mode: {"test": {name: metric for name in suite.SUMMARY_METRICS}}
        for mode in suite.EVALUATION_MODES
    }
    (output / "cross-validation-results.json").write_text(
        json.dumps({"modality_evaluations": evaluations}), encoding="utf-8"
    )
    return {"cv_mean_std": {"folds": 10}}


def test_suite_writes_manifest_summary_and_resumes_only_unfinished_runs(tmp_path: Path) -> None:
    argv = [
        "--study", "missing-modality", "--output-root", str(tmp_path),
        "--run-id", "unit", "--modality-dropout-rates", "0.1", "0.3", "--device", "cpu",
    ]
    calls: list[dict] = []

    def failing_second_run(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return _fake_cross_validation(**kwargs)

    with (
        mock.patch.object(suite, "run_cross_validation", failing_second_run),
        pytest.raises(RuntimeError, match="simulated"),
    ):
        suite.main(argv)
    study_root = tmp_path / "missing-modality" / "unit"
    manifest = json.loads((study_root / "suite.json").read_text(encoding="utf-8"))
    statuses = {key: run["status"] for key, run in manifest["runs"].items()}
    assert statuses == {
        "missing-modality-robustness/no-modality-dropout": "completed",
        "missing-modality-robustness/modality-dropout-0-1": "failed",
    }
    assert calls[1]["config_overrides"]["modality_dropout"] == 0.1

    resumed: list[dict] = []

    def record(**kwargs):
        resumed.append(kwargs)
        return _fake_cross_validation(**kwargs)

    with mock.patch.object(suite, "run_cross_validation", record):
        suite.main(argv)
    # the completed reference arm is not retrained
    assert [call["config_overrides"]["modality_dropout"] for call in resumed] == [0.1, 0.3]
    manifest = json.loads((study_root / "suite.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    with (study_root / "suite-summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["variant"] for row in rows] == [
        "no-modality-dropout", "modality-dropout-0-1", "modality-dropout-0-3",
    ]
    assert float(rows[2]["image_missing.classification_accuracy.mean"]) == 0.3


def test_suite_refuses_to_resume_with_different_settings(tmp_path: Path) -> None:
    argv = ["--study", "ablations", "--output-root", str(tmp_path), "--run-id", "unit"]
    with mock.patch.object(suite, "run_cross_validation", _fake_cross_validation):
        suite.main(argv)
        with pytest.raises(ValueError, match="settings differ"):
            suite.main(argv + ["--epochs", "3"])


def test_deduplication_study_compares_both_settings_with_everything_else_shared(tmp_path: Path) -> None:
    calls: list[dict] = []

    def record(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        passes = 378 if kwargs["dedup_identical_forces"] else 512
        folds = [{"retained_records": passes, "window_counts": {"train": passes * 3, "test": 30}}]
        (output / "cross-validation-results.json").write_text(
            json.dumps({"modality_evaluations": {}, "folds": folds}), encoding="utf-8"
        )
        return {"cv_mean_std": {"folds": 1}}

    argv = ["--study", "deduplication", "--output-root", str(tmp_path), "--run-id", "unit", "--device", "cpu"]
    with mock.patch.object(suite, "run_cross_validation", record):
        suite.main(argv)

    assert [call["dedup_identical_forces"] for call in calls] == [True, False]
    # both arms are scored on the same deduplicated validation/test windows
    assert [call["eval_dedup_identical_forces"] for call in calls] == [True, True]
    # the data setting never reaches ModelConfig, and nothing else differs between arms
    assert all("dedup_identical_forces" not in call["config_overrides"] for call in calls)
    shared = [{k: v for k, v in call.items() if k not in {"dedup_identical_forces", "output_dir"}} for call in calls]
    assert shared[0] == shared[1]
    study_root = tmp_path / "deduplication" / "unit"
    with (study_root / "suite-summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [(row["variant"], row["dedup_identical_forces"], row["retained_passes"]) for row in rows] == [
        ("dedup-true", "True", "378"),
        ("dedup-false", "False", "512"),
    ]
    assert float(rows[1]["train_windows.mean"]) == 1536.0
    assert [row["eval_dedup_identical_forces"] for row in rows] == ["True", "True"]

    # an explicit evaluation setting overrides the study's deduplicated default
    calls.clear()
    with mock.patch.object(suite, "run_cross_validation", record):
        suite.main(["--study", "deduplication", "--output-root", str(tmp_path), "--run-id", "kept",
                    "--device", "cpu", "--eval-dedup-identical-forces", "False"])
    assert [call["eval_dedup_identical_forces"] for call in calls] == [False, False]
    assert len(_experiment_runs("all")) == 31  # the comparison is not part of the paper studies


def test_image_encoder_study_adds_a_frozen_arm_without_the_image_gat() -> None:
    runs = {run["name"]: run["overrides"] for run in suite._experiment_runs("image-encoder")}
    assert list(runs) == ["from-scratch", "pretrained-finetuned", "pretrained-frozen", "pretrained-frozen-no-gat"]
    assert runs["pretrained-frozen-no-gat"] == {"pretrained": True, "freeze_image_encoder": True, "image_gat": False}
    # every other arm keeps the paper's GAT, so only the GAT differs from pretrained-frozen
    assert all(overrides["image_gat"] for name, overrides in runs.items() if name != "pretrained-frozen-no-gat")


def test_fixes_study_changes_one_setting_per_arm_from_the_main_runs() -> None:
    from omnifuser.cli import protocol_kwargs
    from omnifuser.train import model_config

    args = build_parser().parse_args(["--study", "fixes", "--pretrained", "--freeze-image-encoder"])
    common = protocol_kwargs(args)["config_overrides"]
    runs = {run["name"]: run["overrides"] for run in suite._experiment_runs("fixes")}
    assert next(iter(runs)) == "reference" and runs["reference"] == {}

    def config(overrides: dict[str, object]) -> dict[str, object]:
        built = model_config(d_model=32, input_len=96, horizon=96, image_encoder="efficientnet_b0",
                             prediction_head="direct", config_overrides={**common, **overrides})
        return vars(built)

    reference = config({})
    for name, overrides in runs.items():
        changed = {key for key, value in config(overrides).items() if reference[key] != value}
        # an override equal to the protocol's value (e.g. the crop's height of 128) changes nothing
        expected = {key for key, value in overrides.items() if reference[key] != value}
        assert changed == expected and (expected or name == "reference"), name
