"""Launch-retry semantics: crash before Startup Complete is retried, after it the job runs unwatched."""

import os
import sys
from pathlib import Path

import kit_retry

FAKE = Path(__file__).resolve().parent / "fixtures_fake_kit.py"


def run_fake(tmp_path, *, fail_until=0, mode="fail", max_launches=5, window=5.0):
    env = dict(os.environ, FAKE_STATE=str(tmp_path / "state"),
               CERBERUS_KIT_LOG=str(tmp_path / "kit.log"), FAKE_MODE=mode,
               FAKE_FAIL_UNTIL=str(fail_until))
    rows = []
    rc = kit_retry.run([sys.executable, str(FAKE)], max_launches=max_launches, window_s=window, env=env,
                       log=lambda **row: rows.append(row))
    return rc, rows


def test_crashes_are_retried_until_startup(tmp_path):
    rc, rows = run_fake(tmp_path, fail_until=2)
    assert rc == 0
    assert [r["verdict"] for r in rows] == ["LAUNCH_FAIL", "LAUNCH_FAIL", "STARTED"]
    assert "libcarb.tasking/make_fcontext" in rows[0]["note"]
    assert "exit=139" in rows[0]["note"]


def test_a_stall_without_the_marker_is_a_launch_fail(tmp_path):
    rc, rows = run_fake(tmp_path, mode="hang", max_launches=2, window=0.8)
    assert rc == kit_retry.NO_START_EXIT
    assert len(rows) == 2 and all(r["verdict"] == "LAUNCH_FAIL" for r in rows)


def test_success_after_the_marker_is_not_watched(tmp_path):
    rc, rows = run_fake(tmp_path, fail_until=0)
    assert rc == 0 and [r["verdict"] for r in rows] == ["STARTED"]


def test_crash_signature_and_last_line():
    text = "a\n2026 [1ms] [Fatal] libcarb.tasking make_fcontext boom\n"
    assert kit_retry.crash_signature(text) == "libcarb.tasking/make_fcontext"
    assert kit_retry.last_line(text).endswith("boom")
    assert kit_retry.crash_signature("clean") == ""
