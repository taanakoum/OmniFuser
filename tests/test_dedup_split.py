"""Duplicate handling for training versus validation/test data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_alignment_patch import make_nonastreda

from omnifuser.cli import build_parser
from omnifuser.train import train_fold


def _fold(root: Path, out: Path, dedup: bool, eval_dedup: bool | None) -> dict:
    return train_fold(
        data_dir=root, output_dir=out, test_tool="T1", epochs=1, batch_size=2,
        input_len=24, horizon=12, stride=12, device_name="cpu", trim_pass=False,
        image_encoder="compact", max_train_windows=4, num_workers=0,
        dedup_identical_forces=dedup, eval_dedup_identical_forces=eval_dedup,
        config_overrides={"num_rtd_layers": 2, "num_gat_layers": 1, "image_height": 32, "image_width": 96},
    )


def test_deduplicated_evaluation_scores_both_training_settings_on_identical_windows(tmp_path: Path) -> None:
    # 3 tools x 2 runs x 4 blocks = 24 rows; every B4 copies its B3, so 18 distinct passes
    root = make_nonastreda(tmp_path / "data", tools=(1, 2, 3), runs=(1, 2), length=200)
    merged = _fold(root, tmp_path / "merged", dedup=True, eval_dedup=None)
    kept_fair = _fold(root, tmp_path / "kept-fair", dedup=False, eval_dedup=True)
    kept_all = _fold(root, tmp_path / "kept-all", dedup=False, eval_dedup=None)

    assert (merged["retained_records"], merged["evaluation_retained_records"]) == (18, 18)
    assert (kept_fair["retained_records"], kept_fair["evaluation_retained_records"]) == (24, 18)
    assert (kept_all["retained_records"], kept_all["evaluation_retained_records"]) == (24, 24)
    assert merged["eval_dedup_identical_forces"] is True  # follows training when omitted
    assert kept_fair["eval_dedup_identical_forces"] is True
    assert kept_all["eval_dedup_identical_forces"] is False

    def targets(run: Path) -> dict[str, np.ndarray]:
        with np.load(run / "fold-T1-predictions.npz") as data:
            return {key: data[key] for key in ("force_targets", "class_targets")}

    # identical validation/test windows whenever evaluation is deduplicated
    assert merged["window_counts"]["test"] == kept_fair["window_counts"]["test"]
    assert merged["window_counts"]["validation"] == kept_fair["window_counts"]["validation"]
    assert np.array_equal(targets(tmp_path / "merged")["class_targets"], targets(tmp_path / "kept-fair")["class_targets"])
    # the normalizer comes from each run's own training data, so targets match in force units
    assert np.allclose(targets(tmp_path / "merged")["force_targets"], targets(tmp_path / "kept-fair")["force_targets"], atol=1e-3)
    # keeping duplicates for evaluation as well changes the test data
    assert kept_all["window_counts"]["test"] > merged["window_counts"]["test"]
    # only the training data differs between the fair arms
    assert kept_fair["normalizer_mean"] != merged["normalizer_mean"]


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [([], None), (["--eval-dedup-identical-forces", "True"], True), (["--eval-dedup-identical-forces", "false"], False)],
)
def test_eval_dedup_argument_defaults_to_following_training(arguments: list[str], expected: bool | None) -> None:
    import scripts.run_experiment_suite as suite

    assert build_parser().parse_args(["train", *arguments]).eval_dedup_identical_forces is expected
    assert suite.build_parser().parse_args(arguments).eval_dedup_identical_forces is expected


def test_inspect_has_no_evaluation_split() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["inspect", "--eval-dedup-identical-forces", "True"])


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--dedup-identical-forces", "True"], (True, None)),
        (["--dedup-identical-forces", "False", "--eval-dedup-identical-forces", "True"], (False, True)),
        (["--keep-identical-forces"], (False, None)),
        (["--dedup-identical-forces"], (True, None)),
        ([], None),
    ],
)
def test_train_lightning_requires_an_explicit_dedup_setting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    arguments: list[str], expected: tuple[bool, bool | None] | None,
) -> None:
    from scripts import train_lightning

    calls: list[dict] = []
    monkeypatch.setattr(
        train_lightning, "run_cross_validation",
        lambda **kwargs: calls.append(kwargs) or {"cv_mean_std": {}, "out_of_fold": {}},
    )
    monkeypatch.setattr(
        "sys.argv", ["train_lightning.py", "--data-dir", "data", "--output-dir", "out", *arguments]
    )
    if expected is None:
        with pytest.raises(SystemExit) as stopped:
            train_lightning.main()
        assert stopped.value.code == 2
        assert "--dedup-identical-forces True|False" in capsys.readouterr().err
        assert not calls
    else:
        train_lightning.main()
        assert (calls[0]["dedup_identical_forces"], calls[0]["eval_dedup_identical_forces"]) == expected
