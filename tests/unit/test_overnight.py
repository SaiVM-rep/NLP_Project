"""Overnight chain: which experiment runs count as done (rerun cut-off, DECISION-37)."""

from pathlib import Path

from scripts.overnight import finished


def _runs(tmp_path: Path, *names: str) -> Path:
    for n in names:
        (tmp_path / "results" / "E1" / n).mkdir(parents=True)
    return tmp_path


def test_newest_finished_run_ignores_the_running_folder(tmp_path: Path) -> None:
    root = _runs(tmp_path, "20261030T215019-e30656", "20261031T120000-aaaaaa", "_running")
    assert finished("E1", root=root) == root / "results/E1/20261031T120000-aaaaaa"
    assert finished("E5", root=root) is None


def test_since_cutoff_makes_older_runs_not_count(tmp_path: Path) -> None:
    # run ids start with their UTC start time, so a string compare orders them
    root = _runs(tmp_path, "20261030T215019-e30656")
    assert finished("E1", since="20261031T1200", root=root) is None
    _runs(tmp_path, "20261031T130500-bbbbbb")
    got = finished("E1", since="20261031T1200", root=root)
    assert got == root / "results/E1/20261031T130500-bbbbbb"
