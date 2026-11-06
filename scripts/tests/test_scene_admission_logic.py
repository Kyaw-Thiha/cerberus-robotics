import pytest
import scene_admission_logic as sal

from common.cerberus_worlds.admission import STATUS_FAIL, STATUS_PASS, AdmissionRow

TH = sal.Thresholds()


def good_sample(**kw):
    res = {
        "ok": True, "up_axis": "Z", "wall_s": 20.0, "triangles": 1_000_000, "mesh_build_s": 8.0,
        "level_z": 0.0, "footprint_m": [12.0, 9.0], "lidar_hits": 30_000, "lidar_rays": 36_000,
        "spawns": [
            {"x": 1.0, "y": 2.0, "yaw": 0.5, "floor_z": 0.01, "open_radius_m": 1.6},
            {"x": 5.0, "y": 2.0, "yaw": 1.5, "floor_z": 0.01, "open_radius_m": 1.1},
        ],
    }  # fmt: skip
    res.update(kw)
    return res


def good_env(**kw):
    res = {
        "ok": True, "steps_done": 200, "steps_requested": 200, "obs_finite": True, "build_s": 25.0,
        "step_ms_mean": 60.0, "root_z_rel_min": 0.22, "root_z_rel_max": 0.31, "tilt_deg_max": 8.0,
        "drift_m_max": 0.05, "gpu_mem_peak_mib": 6000.0, "gpu_name": "A5000", "shared_scene": True,
    }  # fmt: skip
    res.update(kw)
    return res


def row(**kw):
    args = {
        "sample": good_sample(), "env_small": good_env(), "env_scale": good_env(steps_done=50, steps_requested=50),
        "render": None, "probe": None,
    }  # fmt: skip
    args.update(kw)
    return sal.build_row(
        scene_id="grscenes:a_usd", source="grscenes", category="home", backend="plain_usd",
        th=TH, date_utc="2026-10-06T00:00:00Z", **args,
    )  # fmt: skip


def test_good_scene_passes_with_spawns_and_metrics():
    r = row()
    assert r.status == STATUS_PASS and r.reasons == ()
    assert r.spawn_poses[0].floor_z == 0.01 and len(r.spawn_poses) == 2
    assert r.shared_scene is True
    m = r.metrics
    assert m["build_s_4"] == 25.0 and m["step_ms_16"] == 60.0 and m["peak_gpu_mib"] == 6000.0
    assert m["lidar_hits"] == 30_000 and m["spawn_open_radius_m"] == [1.6, 1.1]
    assert AdmissionRow.from_dict(r.to_dict()) == r  # manifest round trip


@pytest.mark.parametrize(
    "sample, needle",
    [
        (good_sample(spawns=[]), "no valid spawn"),
        (good_sample(up_axis="Y"), "up axis"),
        (good_sample(lidar_hits=100), "lidar hit fraction"),
        (good_sample(lidar_error="RuntimeError('boom')"), "lidar check error"),
        (good_sample(ok=False, error="RuntimeError: no Mesh prims", stage="gather_meshes"), "gather_meshes"),
        ({"timed_out": True, "wall_s": 600.0, "stage": "build_bvh"}, "timeout after 600.0 s at stage 'build_bvh'"),
        ({"returncode": -9}, "without a result"),
    ],
)
def test_sample_failures(sample, needle):
    r = row(sample=sample, env_small=None, env_scale=None)
    assert r.status == STATUS_FAIL
    assert needle in r.reasons[0]
    assert r.metrics.get("build_s_4") is None


def test_sample_failure_without_spawns_keeps_row_valid():
    r = row(sample=good_sample(spawns=[]), env_small=None, env_scale=None)
    assert r.spawn_poses == () and r.status == STATUS_FAIL  # a fail row needs no spawn


@pytest.mark.parametrize(
    "env, needle",
    [
        (good_env(root_z_rel_min=-0.73, root_z_rel_max=-0.49), "fell through"),  # the Galileo case
        (good_env(root_z_rel_max=0.9), "> 0.5"),
        (good_env(tilt_deg_max=45.0), "tilt"),
        (good_env(drift_m_max=3.0), "drift"),
        (good_env(obs_finite=False), "non-finite"),
        (good_env(steps_done=37), "37/200"),
        (good_env(root_z_rel_min=None, root_z_rel_max=None), "no root height"),
        (good_env(root_z_rel_min=float("nan"), root_z_rel_max=float("nan")), "fell through"),
        ({"timed_out": True, "wall_s": 480.0, "stage": "build_env"}, "env4: timeout after 480.0 s at stage 'build_env'"),
        ({"ok": False, "error": "OutOfMemory\nstack", "stage": "build_env"}, "env4: error at stage 'build_env'"),
    ],
)
def test_env4_failures(env, needle):
    r = row(env_small=env, env_scale=None)
    assert r.status == STATUS_FAIL
    assert any(needle in x for x in r.reasons), r.reasons
    assert r.spawn_poses  # failed rows keep their spawns for debugging


def test_scale_run_limits_step_time_and_memory_and_timeouts():
    slow = row(env_scale=good_env(step_ms_mean=900.0))
    assert slow.status == STATUS_FAIL and "env16: step 900.0 ms" in slow.reasons[0]
    big = row(env_scale=good_env(gpu_mem_peak_mib=23_000.0))
    assert "env16: peak GPU" in big.reasons[0]
    hung = row(env_scale={"timed_out": True, "wall_s": 480.0, "stage": "build_env"})
    assert "env16: timeout" in hung.reasons[0]
    # the 4-env run is not held to the scale limits
    assert row(env_small=good_env(step_ms_mean=900.0, gpu_mem_peak_mib=23_000.0)).status == STATUS_PASS


def test_combined_env_row_uses_one_launch_and_omits_split_metrics():
    """--combined-env-phases: one run answers both the stability and the scale rules."""
    r = row(env_small=None, env_scale=None, env_combined=good_env())
    assert r.status == "pass" and r.reasons == ()
    assert r.metrics["env_phase"] == "combined"
    assert r.metrics["build_s"] == 25.0 and r.metrics["step_ms"] == 60.0
    assert "build_s_4" not in r.metrics and "build_s_16" not in r.metrics  # never claims a 4-env build
    assert r.metrics["tilt_deg"] == 8.0  # shared metrics still come from the one run


def test_combined_env_row_fails_on_a_stability_rule_and_on_a_scale_rule():
    stability = row(env_small=None, env_scale=None, env_combined=good_env(root_z_rel_min=0.01))
    assert stability.status == "fail" and any("base z" in x for x in stability.reasons)
    scale = row(env_small=None, env_scale=None, env_combined=good_env(step_ms_mean=900.0))
    assert scale.status == "fail" and any("step" in x for x in scale.reasons)


def test_split_rows_still_report_split_metrics():
    r = row()
    assert r.status == "pass" and r.metrics["env_phase"] == "split"
    assert r.metrics["build_s_4"] == 25.0 and r.metrics["build_s_16"] == 25.0
    assert "build_s" not in r.metrics


def test_later_phases_are_not_blamed_when_an_earlier_one_failed():
    status, reasons = sal.evaluate_scene(good_sample(spawns=[]), None, None, None, TH)
    assert status == STATUS_FAIL and len(reasons) == 1
    status, reasons = sal.evaluate_scene(good_sample(), good_env(tilt_deg_max=50.0), None, None, TH)
    assert len(reasons) == 1 and "env4" in reasons[0]
    status, reasons = sal.evaluate_scene(good_sample(), None, None, None, TH)
    assert reasons == ["internal: phase not run"]  # a missing phase after a pass is a bug, not a pass


def test_render_is_only_required_when_asked_for():
    assert row(render={"ok": True, "frame_std": 0.0}).status == STATUS_PASS
    status, reasons = sal.evaluate_scene(
        good_sample(), good_env(), good_env(), {"ok": True, "frame_std": 0.1}, TH, render_required=True
    )
    assert status == STATUS_FAIL and "blank" in reasons[0]
    status, _ = sal.evaluate_scene(
        good_sample(), good_env(), good_env(), {"ok": True, "frame_std": 40.0}, TH, render_required=True
    )
    assert status == STATUS_PASS
    status, reasons = sal.evaluate_scene(good_sample(), good_env(), good_env(), None, TH, render_required=True)
    assert "render phase not run" in reasons[0]


def test_arena_probe_is_recorded_and_never_changes_the_verdict():
    bad = sal.summarize_probe({"timed_out": True, "wall_s": 480.0, "stage": "build_env"}, TH)
    assert bad["status"] == STATUS_FAIL and bad["timed_out"] and "timeout" in bad["reasons"][0]
    ok = sal.summarize_probe(good_env(steps_done=50, steps_requested=50), TH)
    assert ok["status"] == STATUS_PASS and ok["build_s"] == 25.0
    r = row(probe=bad)
    assert r.status == STATUS_PASS and r.metrics["arena_probe"]["status"] == STATUS_FAIL


def test_shared_scene_mode_follows_the_env_run():
    assert row(env_small=good_env(shared_scene=False)).shared_scene is False


def test_skip_row():
    r = sal.skip_row(scene_id="grscenes:x", source="grscenes", category="home", backend="plain_usd", reason="missing", date_utc="t")
    assert r.status == "skip" and r.reasons == ("missing",)


def test_thresholds_are_frozen_and_documented_as_dict():
    with pytest.raises(AttributeError):
        TH.max_tilt_deg = 99.0
    assert sal.thresholds_dict(TH)["clear_height_m"] == 0.8


# ---- Lightwheel subset ----------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 5, 10, 12, 37, 100])
def test_kitchen_subset_is_distinct_deterministic_and_in_range(n):
    pairs = sal.stratified_kitchen_subset(n)
    assert len(pairs) == len(set(pairs)) == n
    assert pairs == sal.stratified_kitchen_subset(n)
    assert all(1 <= layout <= 10 and 1 <= style <= 10 for layout, style in pairs)


def test_kitchen_subset_spreads_over_layouts_and_styles():
    pairs = sal.stratified_kitchen_subset(12)
    assert pairs[0] == (1, 1)  # the kitchen known to download anonymously
    assert {p[0] for p in pairs} == set(range(1, 11))
    assert {p[1] for p in pairs} == set(range(1, 11))
    assert len(set(sal.stratified_kitchen_subset(10))) == 10
    assert {p[0] for p in sal.stratified_kitchen_subset(10)} == set(range(1, 11))
    assert len(set(sal.stratified_kitchen_subset(100))) == 100


@pytest.mark.parametrize("n", [0, 101, -1])
def test_kitchen_subset_rejects_bad_sizes(n):
    with pytest.raises(ValueError):
        sal.stratified_kitchen_subset(n)


# ---- resume / limit ---------------------------------------------------------------------------------------
def _existing(statuses):
    return [
        AdmissionRow(
            scene_id=sid, source="grscenes", status=st,
            spawn_poses=(sal.SpawnPose(0.0, 0.0, 0.0, 0.0),) if st == STATUS_PASS else (),
        )
        for sid, st in statuses.items()
    ]  # fmt: skip


SCENES = [("h1", "home"), ("h2", "home"), ("h3", "home"), ("c1", "commercial"), ("c2", "commercial"), ("c3", "commercial")]


def test_without_resume_everything_runs():
    assert sal.scenes_to_run(SCENES, _existing({"h1": "pass"}), resume=False) == [s for s, _ in SCENES]


def test_resume_skips_everything_in_the_manifest_and_retry_failed_reruns_fails():
    existing = _existing({"h1": "pass", "h2": "fail", "c1": "skip"})
    assert sal.scenes_to_run(SCENES, existing, resume=True) == ["h3", "c2", "c3"]
    assert sal.scenes_to_run(SCENES, existing, resume=True, retry_failed=True) == ["h2", "h3", "c2", "c3"]


def test_limit_is_per_category_and_counted_after_resume_skips():
    assert sal.scenes_to_run(SCENES, [], resume=False, limit=2) == ["h1", "h2", "c1", "c2"]
    existing = _existing({"h1": "pass", "c1": "fail"})
    assert sal.scenes_to_run(SCENES, existing, resume=True, limit=2) == ["h2", "h3", "c2", "c3"]
    assert sal.scenes_to_run(SCENES, [], resume=False, limit=1) == ["h1", "c1"]
