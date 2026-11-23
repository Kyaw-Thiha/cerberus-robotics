"""The relaunch rule: units decide success, never the exit code. Unit tests use a fake job that crashes at random.

Evidence for the rule: Kit's native exit(0) after a long sim block and the loader SIGSEGV both reach the caller as
exit 0 (AppLauncher), so a "clean" run without its outputs is a crash.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HOST = Path(__file__).resolve().parents[1]
WRAPPER = HOST / "run_until_complete.py"
FAKE = HOST / "tests" / "fake_crashing_job.py"

sys.path.insert(0, str(HOST))
sys.path.insert(0, str(HOST.parents[1]))
import run_until_complete as ruc
from common.cerberus_eval import atomic_io


def run_fake(tmp_path: Path, *, total: int, crash_p: float, seed: int = 0, max_launches: int = 60):
    units = tmp_path / "units"
    log_dir = tmp_path / "logs"
    cmd = [sys.executable, str(WRAPPER), "--name", "fake", "--units-dir", str(units),
           "--expect", str(total), "--glob", "unit_*.json", "--max-launches", str(max_launches),
           "--log-dir", str(log_dir), "--",
           sys.executable, str(FAKE), "--units-dir", str(units), "--total", str(total),
           "--crash-p", str(crash_p), "--seed", str(seed)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc, units, log_dir


def read_rows(log_dir: Path):
    import csv
    rows = list(csv.DictReader((log_dir / "fake.csv").open(encoding="utf-8")))
    return rows


def test_random_crashes_still_finish_every_unit(tmp_path):
    proc, units, log_dir = run_fake(tmp_path, total=8, crash_p=0.5, seed=3)
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in units.glob("unit_*.json")) == [f"unit_{i:02d}.json" for i in range(8)]
    rows = read_rows(log_dir)
    assert rows[-1]["verdict"] == "DONE"
    assert int(rows[-1]["units"]) == 8
    # the fake job crashed at least sometimes; every non-final launch that stopped short is recorded as a crash
    assert len(rows) > 1
    for row in rows[:-1]:
        assert row["verdict"] in {"CRASH", "CRASH_NO_MARKER", "SHORT"}


def test_clean_exit_without_marker_is_a_crash(tmp_path):
    """A job that exits 0 but writes nothing must be relaunched, not accepted."""
    proc, units, log_dir = run_fake(tmp_path, total=4, crash_p=1.0, max_launches=8)
    assert proc.returncode == ruc.EXHAUSTED_EXIT
    rows = read_rows(log_dir)
    verdicts = [row["verdict"] for row in rows]
    assert set(verdicts) <= {"CRASH", "CRASH_NO_MARKER"}
    assert "CRASH_NO_MARKER" in verdicts, verdicts
    assert not list(units.glob("unit_*.json"))


def test_an_incomplete_unit_is_not_counted(tmp_path):
    units = tmp_path / "u"
    units.mkdir()
    (units / "a.json").write_text('{"ok": true')
    (units / "b.json").write_text("")
    (units / "c.json").write_text(json.dumps({"ok": True}))
    assert ruc.count_units(units, "*.json") == 1


def test_atomic_write_leaves_no_partial_file(tmp_path):
    target = tmp_path / "deep" / "unit.json"
    atomic_io.write_json_atomic(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}
    assert not list(target.parent.glob("*.part"))


def test_resume_uses_the_newest_checkpoint(tmp_path):
    """Launch 2 gets --resume <newest checkpoint>; launch 1 does not, so a fresh start does not need one."""
    units = tmp_path / "ckpt"
    units.mkdir()
    log_dir = tmp_path / "logs"
    record = tmp_path / "argv.jsonl"
    job = tmp_path / "job.py"
    job.write_text(
        "import json, pathlib, sys\n"
        f"open({str(record)!r},'a').write(json.dumps(sys.argv[1:])+chr(10))\n"
        f"d = pathlib.Path({str(units)!r})\n"
        "n = len(list(d.glob('model_*.pt')))\n"
        "(d / f'model_{n+1:02d}.pt').write_bytes(b'ckpt')\n"
        "sys.exit(0)\n"
    )
    cmd = [sys.executable, str(WRAPPER), "--name", "train", "--units-dir", str(units),
           "--expect", "2", "--glob", "model_*.pt", "--max-launches", "3",
           "--resume-glob", "model_*.pt", "--log-dir", str(log_dir),
           "--", sys.executable, str(job)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    assert len(calls) == 2
    assert calls[0] == []
    assert calls[1][0] == "--resume" and calls[1][1].endswith("model_01.pt")


def test_newest_checkpoint_by_mtime(tmp_path):
    units = tmp_path / "ckpt"
    units.mkdir()
    import os, time
    (units / "model_10.pt").write_bytes(b"x")
    (units / "model_20.pt").write_bytes(b"y")
    os.utime(units / "model_10.pt", (1000, 1000))
    os.utime(units / "model_20.pt", (time.time(), time.time()))
    assert ruc.newest_checkpoint(units, "model_*.pt").name == "model_20.pt"
