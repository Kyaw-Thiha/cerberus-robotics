"""Tests for the Kit log capture and the replay selection."""


import kit_logs
from nav_eval import select_replay_indices

# --- Kit log capture ---------------------------------------------------------------------------

SAMPLE_LOG = """\
2026-11-13T10:00:01Z [0ms] [Info] App Starting
2026-11-13T10:00:02Z [1ms] [Info] loading extensions
2026-11-13T10:00:05Z [4ms] [Info] waiting for compilation of ray tracing shaders
2026-11-13T10:03:00Z [3ms] [Info] App Ready
2026-11-13T10:03:01Z [1ms] [Info] Startup Complete
"""


def test_timeline_markers_and_elapsed():
    timeline = kit_logs.parse_timeline(SAMPLE_LOG)
    markers = timeline["markers"]
    assert set(markers) == {"app_starting", "raytracing_shader_compile", "app_ready", "startup_complete"}
    assert markers["app_starting"]["elapsed_s"] == 0.0
    assert markers["startup_complete"]["elapsed_s"] == 180.0


def test_copy_kit_log_finds_and_copies_the_newest(tmp_path):
    root = tmp_path / "logs"
    (root / "Kit").mkdir(parents=True)
    old = root / "Kit" / "kit_old.log"
    new = root / "Kit" / "kit_new.log"
    old.write_text("old\n")
    new.write_text(SAMPLE_LOG)
    out = tmp_path / "out"
    timeline = kit_logs.copy_kit_log(out, stage="probe", stdout_text="hello", extra_roots=[root])
    assert timeline["kit_log_source"] == str(new)
    copied = list(out.rglob("kit_new.log"))
    assert copied and copied[0].read_text() == SAMPLE_LOG
    assert next(iter(out.glob("*/launch_timeline.json"))).is_file()
    assert timeline["markers"]


# --- replay selection --------------------------------------------------------------------------

def test_replay_selection_by_outcome():
    records = [{"outcome": "fall"}, {"outcome": "success"}, {"outcome": "timeout"}, {"outcome": "success"}]
    assert select_replay_indices(records, "", "success,failure") == [1, 0]
    assert select_replay_indices(records, "3,1", "success,failure") == [3, 1]
    assert select_replay_indices(records, "", "failure") == [0]
