"""run_resilient relaunches a crashed resumable job without duplicating or losing work."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def run(out: Path, *, crash_after: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(REPO / "scripts" / "host" / "run_resilient.sh"), "--name", "toy", "--max-restarts", "3", "--",
         sys.executable, str(REPO / "scripts" / "host" / "tests" / "fixtures_resumable_job.py"),
         "--out", str(out), "--pairs", "6", "--crash-after", str(crash_after)],
        cwd=str(REPO), capture_output=True, text=True, timeout=120, check=False)


def test_crash_and_relaunch_loses_and_duplicates_nothing(tmp_path):
    crashed = tmp_path / "crashed"
    result = run(crashed, crash_after=3)
    assert result.returncode == 0, result.stderr
    assert "attempt 2" in result.stderr  # it really was relaunched

    clean = tmp_path / "clean"
    assert run(clean, crash_after=99).returncode == 0

    crashed_files = sorted((crashed / "pairs").glob("pair_*.json"))
    clean_files = sorted((clean / "pairs").glob("pair_*.json"))
    assert [p.name for p in crashed_files] == [p.name for p in clean_files]
    assert len(crashed_files) == 6
    for a, b in zip(crashed_files, clean_files):
        assert json.loads(a.read_text()) == json.loads(b.read_text())
