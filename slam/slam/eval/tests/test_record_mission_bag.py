import threading
import time
from pathlib import Path

import pytest

from slam.slam.eval import record_mission_bag as rmb


def test_run_dir_for_builds_backend_and_timestamp_path():
    assert rmb.run_dir_for("fast_lio2", "2026-09-24_10-45-32") == Path("slam/slam/eval/results/fast_lio2/2026-09-24_10-45-32")


def test_record_until_sim_done_outlasts_a_slow_sim_then_stops_on_the_sentinel(tmp_path, monkeypatch):
    """The sim runs slower than real time: the recorder must keep going past the mission length until the sentinel."""
    monkeypatch.setattr(rmb, "DRAIN_SECONDS", 0.2)
    done = tmp_path / ".sim_exited"
    threading.Timer(2.5, done.touch).start()
    start = time.monotonic()
    assert rmb.record_until_sim_done(["sleep", "60"], done, run_timeout_s=30.0) is True
    assert 2.5 <= time.monotonic() - start < 10.0


def test_record_until_sim_done_gives_up_at_the_hard_cap(tmp_path):
    start = time.monotonic()
    assert rmb.record_until_sim_done(["sleep", "60"], tmp_path / "never", run_timeout_s=1.5) is False
    assert time.monotonic() - start < 10.0


def test_main_requires_the_sim_done_file(monkeypatch):
    monkeypatch.setattr("sys.argv", ["x", "--backend", "b", "--mission", "household", "--run-timestamp", "t", "--sensor-rig", "lidar_camera"])
    with pytest.raises(SystemExit) as exc:
        rmb.main()
    assert exc.value.code == 2
