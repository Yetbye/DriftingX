"""Run-directory stamping and checkpoint-based resume.

These guard the offline-cluster failure mode where a resubmitted job silently
starts a fresh run instead of resuming.
"""

from pathlib import Path

import pytest

from utils.misc import stamp_workdir


def test_bare_runs_gets_timestamped(tmp_path):
    out = stamp_workdir(str(tmp_path / "runs"))
    assert Path(out).parent == tmp_path / "runs"
    assert out != str(tmp_path / "runs")
    assert Path(out).name[:4].isdigit()


def test_named_dir_gets_prefixed(tmp_path):
    out = stamp_workdir(str(tmp_path / "my_run"))
    assert Path(out).name.endswith("_my_run")
    assert Path(out).name[:4].isdigit()


def test_already_stamped_name_is_reused(tmp_path):
    stamped = tmp_path / "0917_0751_crossflow_B"
    assert stamp_workdir(str(stamped)) == str(stamped)


def test_no_stamp_returns_verbatim(tmp_path):
    target = tmp_path / "crossflow_B"
    assert stamp_workdir(str(target), no_stamp=True) == str(target)


def test_resume_requires_existing_checkpoint(tmp_path):
    target = tmp_path / "run"
    (target / "checkpoints").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        stamp_workdir(str(target), resume=True)
    # once a checkpoint exists, resume returns the path verbatim
    (target / "checkpoints" / "step_000000004.pt").write_bytes(b"x")
    assert stamp_workdir(str(target), resume=True) == str(target)


def test_resume_never_double_stamps(tmp_path):
    """The bug this test pins: resume must not produce <newstamp>_<oldstamp>_name."""
    target = tmp_path / "0917_0751_run"
    (target / "checkpoints").mkdir(parents=True)
    (target / "checkpoints" / "step_000000002.pt").write_bytes(b"x")
    assert stamp_workdir(str(target), resume=True) == str(target)
    assert stamp_workdir(str(target)) == str(target)
