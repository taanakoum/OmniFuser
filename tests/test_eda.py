import json
from pathlib import Path

import numpy as np
import pytest
from test_alignment_patch import make_nonastreda

from scripts.eda import autocorrelation, label_dynamics, run_eda, trimming_diagnostics


def summary_rows(out: Path) -> list[dict]:
    import csv

    with (out / "difficulty_by_time_step.csv").open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_trimming_flags_an_idle_reference_window() -> None:
    cutting = np.full((10_000, 3), 50.0)
    clean = cutting.copy()
    clean[:200] = 0.5  # 2 % idle lead-in: reference window is mostly cutting
    late_start = cutting.copy()
    late_start[:400] = 0.5  # 4 % idle lead-in: reference median is the idle level

    good = trimming_diagnostics(clean)
    bad = trimming_diagnostics(late_start)

    assert not good["reference_is_idle"] and good["trimmed_lead_samples"] >= 190
    assert good["residual_lead_idle_samples"] < 20
    assert bad["reference_is_idle"]
    assert bad["trimmed_lead_samples"] == 0  # the loader keeps this lead-in
    assert bad["residual_lead_idle_samples"] >= 390
    assert bad["alternative_trim_lead_samples"] >= 390


def test_label_dynamics_counts_mixed_runs_and_backward_flips() -> None:
    rows = [
        {"tool": "T1", "run": 1, "block": 1, "image_label": "sharp", "flank_wear": 10.0},
        {"tool": "T1", "run": 1, "block": 2, "image_label": "used", "flank_wear": 20.0},
        {"tool": "T1", "run": 2, "block": 1, "image_label": "sharp", "flank_wear": 15.0},
        {"tool": "T1", "run": 2, "block": 2, "image_label": "sharp", "flank_wear": 16.0},
    ]
    summary, runs = label_dynamics(rows, "raw")
    assert summary["runs"] == 2 and summary["runs_with_mixed_classes"] == 1
    assert summary["backward_class_flips"] == 1 and summary["forward_class_flips"] == 1
    assert summary["flank_wear_decreases_along_tool_order"] == 1
    assert [run["mixed_classes"] for run in runs] == [True, False]


def test_autocorrelation_of_a_slow_sine_decays_then_turns_negative() -> None:
    values = autocorrelation(np.sin(np.arange(400) * 2 * np.pi / 40), 20)
    assert values[0] == 1.0 and values[5] > 0.6 and values[20] < -0.8


def test_eda_runs_end_to_end_and_writes_every_output(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    root = make_nonastreda(tmp_path / "data", tools=(1, 2, 3, 4), runs=(1, 2), length=700)
    out = tmp_path / "eda"

    summary = run_eda(root, out, pool=2, input_len=24, horizon=24, stride=12,
                      difficulty_pools=(1, 2), windows_per_tool=50)

    assert summary["inventory"]["raw_records"] == 32 and summary["inventory"]["tools"] == 4
    # the fixture copies B3 into B4 in every run
    assert summary["duplicates"]["records_removed"] == 8
    # fixture label = (run + block) % 3; B4 outranks its B3 copy, so blocks 1, 2, 4 remain:
    # run 1 -> dulled, sharp, dulled; run 2 -> sharp, used, sharp; four tools
    assert summary["labels"]["class_counts"]["retained"] == {"sharp": 12, "used": 4, "dulled": 8}
    assert summary["protocol_windows"]["evaluation_stride"] == 24
    for table in ("records.csv", "runs.csv", "tools.csv", "folds.csv",
                  "difficulty_by_time_step.csv", "autocorrelation.csv"):
        assert (out / table).is_file(), table
    assert len(summary["figures"]) == 12
    assert all((out / figure).is_file() for figure in summary["figures"])
    saved = json.loads((out / "eda_summary.json").read_text(encoding="utf-8"))
    alignment = saved["persistence_and_alignment"]
    assert all(0.0 <= value <= 1.0 for value in alignment.values())
    # every tool contributes one difficulty row per pool and paper horizon that fits
    assert {row["pool"] for row in summary_rows(out)} == {"1", "2"}
    folds = (out / "folds.csv").read_text(encoding="utf-8").splitlines()
    assert len(folds) == 5  # header + 4 leave-one-tool-out folds
