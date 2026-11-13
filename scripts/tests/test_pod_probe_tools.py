"""Tests for the liveness probe, Kit log capture, the cache bundle, the budget and the replay selection."""

from pathlib import Path

import pytest

import kit_cache
import kit_logs
import pod_budget
import pod_launch_probe as probe
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
    assert (list(out.glob("*/launch_timeline.json"))[0]).is_file()
    assert timeline["markers"]


# --- liveness probe ----------------------------------------------------------------------------

def samples(*rows):
    return [probe.Sample(elapsed_s=e, log_size=s, cpu_pct=c, gpu_util=g) for e, s, c, g in rows]


def test_growth_means_not_hung():
    rows = samples((0, 100, 1, 0), (60, 200, 1, 0), (120, 300, 1, 0), (180, 400, 1, 0), (240, 500, 1, 0))
    assert not probe.hang_detected(rows)


def test_flat_log_with_idle_cpu_and_gpu_is_hung():
    rows = samples((0, 100, 1, 0), (60, 100, 1, 0), (120, 100, 1, 0), (180, 100, 1, 0), (240, 100, 1, 0))
    assert probe.hang_detected(rows)


def test_busy_cpu_or_gpu_is_not_hung():
    assert not probe.hang_detected(samples((0, 100, 80, 0), (240, 100, 80, 0)))
    assert not probe.hang_detected(samples((0, 100, 1, 90), (240, 100, 1, 90)))


def test_window_must_span_four_minutes():
    rows = samples((0, 100, 1, 0), (60, 100, 1, 0), (120, 100, 1, 0))
    assert not probe.hang_detected(rows)


def test_verdicts():
    ok = probe.decide(exited=True, rc=0, output="PROBE_OK", elapsed_s=60, killed_for_hang=False, samples=[])
    slow = probe.decide(exited=True, rc=0, output="PROBE_OK", elapsed_s=600, killed_for_hang=False, samples=[])
    assert (ok, slow) == ("OK", "SLOW_OK")

    segv = probe.decide(exited=True, rc=139, output="", elapsed_s=7, killed_for_hang=False, samples=[])
    wrapped = probe.decide(exited=True, rc=1, output="terminating this process with exit code 139.",
                           elapsed_s=7, killed_for_hang=False, samples=[])
    other = probe.decide(exited=True, rc=1, output="boom", elapsed_s=7, killed_for_hang=False, samples=[])
    assert (segv, wrapped, other) == ("SEGFAULT", "SEGFAULT", "OTHER")

    hung = probe.decide(exited=False, rc=None, output="", elapsed_s=300, killed_for_hang=True, samples=[])
    assert hung == "HANG"

    busy = samples((0, 100, 50, 0), (120, 200, 50, 0))
    active = probe.decide(exited=False, rc=None, output="", elapsed_s=1200, killed_for_hang=False, samples=busy)
    idle = probe.decide(exited=False, rc=None, output="", elapsed_s=1200, killed_for_hang=False,
                        samples=samples((0, 100, 1, 0), (240, 100, 1, 0)))
    assert active == "TIMEOUT_WITH_ACTIVITY"
    assert idle == "HANG"


def test_only_segfault_and_hang_count_as_host_failures(tmp_path):
    path = tmp_path / "launch_probe.csv"
    rows = [
        {"ts": "", "stage": "probe", "driver": "1", "cuda": "", "gpu": "", "host": "a", "datacenter": "",
         "image": "", "verdict": "SLOW_OK", "tier": "", "cache": "", "note": ""},
        {"ts": "", "stage": "probe", "driver": "1", "cuda": "", "gpu": "", "host": "b", "datacenter": "",
         "image": "", "verdict": "TIMEOUT_WITH_ACTIVITY", "tier": "", "cache": "", "note": ""},
        {"ts": "", "stage": "probe", "driver": "1", "cuda": "", "gpu": "", "host": "c", "datacenter": "",
         "image": "", "verdict": "SEGFAULT", "tier": "", "cache": "", "note": ""},
        {"ts": "", "stage": "probe", "driver": "1", "cuda": "", "gpu": "", "host": "d", "datacenter": "",
         "image": "", "verdict": "HANG", "tier": "", "cache": "", "note": ""},
    ]
    import csv

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert probe.non_ok_hosts(path) == {"c", "d"}


# --- cache bundle ------------------------------------------------------------------------------

def make_home(tmp_path):
    home = tmp_path / "home"
    (home / ".cache" / "ov").mkdir(parents=True)
    (home / ".cache" / "ov" / "shader.bin").write_text("x" * 1024)
    (home / ".cache" / "warp").mkdir(parents=True)
    (home / ".cache" / "warp" / "w.bin").write_text("y" * 512)
    return home


def test_discover_finds_only_non_empty_candidates(tmp_path):
    home = make_home(tmp_path)
    (home / ".cache" / "kit").mkdir()  # empty: ignored
    found = {p.name for p in kit_cache.discover(home)}
    assert found == {"ov", "warp"}


def test_save_and_restore_round_trip(tmp_path):
    source_home = make_home(tmp_path)
    work = tmp_path / "work"
    uploaded = {}

    def upload(archive: Path, remote_dir: str) -> bool:
        uploaded["archive"] = archive
        uploaded["remote"] = remote_dir
        return True

    manifest = kit_cache.save("595.91.07", "NVIDIA GeForce RTX 4090", home=source_home,
                              upload=upload, out_dir=work)
    assert manifest["saved"] and manifest["dirs"] == [".cache/ov", ".cache/warp"]
    assert uploaded["remote"] == "cache/kit/595.91.07_nvidia_geforce_rtx_4090"

    dest_home = tmp_path / "dest"
    dest_home.mkdir()
    restore_work = tmp_path / "restore"

    def download(remote_dir: str, local_dir: Path) -> bool:
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / uploaded["archive"].name).write_bytes(uploaded["archive"].read_bytes())
        return True

    result = kit_cache.restore("595.91.07", "NVIDIA GeForce RTX 4090", home=dest_home,
                               download=download, work_dir=restore_work)
    assert result["restored"]
    assert (dest_home / ".cache" / "ov" / "shader.bin").is_file()
    assert kit_cache.cache_restored("595.91.07", "NVIDIA GeForce RTX 4090", home=dest_home)
    assert not kit_cache.cache_restored("580.95.05", "NVIDIA GeForce RTX 4090", home=dest_home)


def test_restore_refuses_a_missing_bundle(tmp_path):
    result = kit_cache.restore("595.91.07", "GPU", home=tmp_path, download=lambda *a: False,
                               work_dir=tmp_path / "w")
    assert not result["restored"]


def test_cache_key_is_driver_and_gpu_specific():
    assert kit_cache.cache_key("595.91.07", "NVIDIA GeForce RTX 4090") != \
        kit_cache.cache_key("580.95.05", "NVIDIA GeForce RTX 4090")


# --- budget and replay -------------------------------------------------------------------------

def test_spend_estimate_and_warning():
    est = pod_budget.estimate(0.89, started_epoch=0, now_epoch=3600)
    assert est["elapsed_min"] == 60.0 and est["usd"] == pytest.approx(0.89)
    assert not est["over_warn"]
    assert pod_budget.estimate(1.0, 0, 6 * 3600)["over_warn"]


def test_stage_caps_cover_the_documented_stages():
    for stage in ("provision", "fetch", "grids", "stand", "navigation", "wrap_up"):
        assert stage in pod_budget.STAGE_CAPS_MIN


def test_replay_selection_by_outcome():
    records = [{"outcome": "fall"}, {"outcome": "success"}, {"outcome": "timeout"}, {"outcome": "success"}]
    assert select_replay_indices(records, "", "success,failure") == [1, 0]
    assert select_replay_indices(records, "3,1", "success,failure") == [3, 1]
    assert select_replay_indices(records, "", "failure") == [0]
