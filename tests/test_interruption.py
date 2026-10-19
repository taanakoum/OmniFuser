"""Behaviour when a job is killed mid-run, e.g. at a cluster time limit."""
from __future__ import annotations

from pathlib import Path

import pytest
from test_alignment_patch import make_nonastreda

import omnifuser.train as train_module
from omnifuser.cross_validation import _load_finished_fold
from omnifuser.train import train_fold

TINY = {"num_rtd_layers": 2, "num_gat_layers": 1, "image_height": 32, "image_width": 96}


def _fold(root: Path, out: Path) -> dict:
    return train_fold(
        data_dir=root, output_dir=out, test_tool="T1", epochs=2, batch_size=2,
        input_len=24, horizon=12, stride=12, device_name="cpu", trim_pass=False,
        image_encoder="compact", max_train_windows=4, num_workers=0,
        config_overrides=TINY,
    )


def test_truncated_resume_file_restarts_the_fold(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = make_nonastreda(tmp_path / "data", tools=(1, 2, 3), runs=(1,), length=200)
    out = tmp_path / "run"
    out.mkdir()
    (out / "fold-T1-resume.pt").write_bytes(b"PK\x03\x04 truncated")

    result = _fold(root, out)

    assert "ignoring an unreadable resume file" in capsys.readouterr().out
    assert [entry["epoch"] for entry in result["history"]] == [1, 2]
    assert not (out / "fold-T1-resume.pt").exists()
    assert not list(out.glob("*.tmp"))


def test_fold_is_marked_finished_only_after_every_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_nonastreda(tmp_path / "data", tools=(1, 2, 3), runs=(1,), length=200)
    out = tmp_path / "run"
    real_save = train_module._atomic_torch_save

    def killed_at_final_checkpoint(payload: dict, path: Path) -> None:
        if path.name == "fold-T1.pt":
            raise KeyboardInterrupt("job killed")
        real_save(payload, path)

    monkeypatch.setattr(train_module, "_atomic_torch_save", killed_at_final_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        _fold(root, out)

    # no completion marker, so a rerun retrains this fold instead of reusing it
    assert not (out / "fold-T1.json").exists()
    assert _load_finished_fold(out, "T1", {}, {}, 2, 2) is None
    # the per-epoch resume file survived and is readable
    assert (out / "fold-T1-resume.pt").is_file()

    monkeypatch.setattr(train_module, "_atomic_torch_save", real_save)
    result = _fold(root, out)
    assert (out / "fold-T1.json").is_file() and (out / "fold-T1.pt").is_file()
    assert [entry["epoch"] for entry in result["history"]] == [1, 2]
