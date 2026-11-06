import json
import sys

import pytest
import validate_scenes as vs
from scene_admission_logic import Thresholds
from test_scene_admission_logic import good_env, good_sample

from common.cerberus_worlds import admission

TH = Thresholds()


def parse(*argv):
    return vs.build_parser().parse_args(list(argv))


@pytest.fixture
def scenes_dir(tmp_path, monkeypatch):
    """data/scenes layout with home scenes of 300/100/200 bytes and commercial scenes of 50/400 bytes."""
    monkeypatch.setenv("CERBERUS_SCENES_DIR", str(tmp_path))
    for category, sizes in {"home": {"h_big_usd": 300, "h_small_usd": 100, "h_mid_usd": 200}, "commercial": {"c_a_usd": 50, "c_b_usd": 400}}.items():
        for sid, size in sizes.items():
            folder = tmp_path / "grscenes" / category / "scenes" / sid
            folder.mkdir(parents=True)
            (folder / "start_result_navigation.usd").write_bytes(b"#usda 1.0\n(\n    metersPerUnit = 1\n)\n#" + b"x" * size)
    return tmp_path


# ---- arguments ---------------------------------------------------------------------------------------------
def test_defaults_match_the_spec():
    a = parse()
    assert (a.source, a.num_envs, a.scale_envs, a.steps, a.build_timeout_s) == ("grscenes", 4, 16, 200, 900.0)
    assert a.categories == "home,commercial" and not a.resume and not a.render and a.arena_probe == 0
    assert a.limit is None and a.watchdog_s > 0 and a.child is None


@pytest.mark.parametrize(
    "argv",
    [
        ["--num-envs", "0"], ["--steps", "0"], ["--limit", "0"], ["--arena-probe", "-1"],
        ["--categories", "home,garden"], ["--retry-failed"], ["--build-timeout-s", "0"],
        ["--source", "lightwheel", "--lightwheel-subset", "0"], ["--child", "env"],
    ],
)  # fmt: skip
def test_invalid_arguments_exit_2(argv, capsys):
    assert vs.main(argv) == 2
    assert "[ERROR]" in capsys.readouterr().err


def test_unknown_source_is_an_argparse_error():
    with pytest.raises(SystemExit) as e:
        vs.main(["--source", "nowhere"])
    assert e.value.code == 2


def test_child_arguments_round_trip():
    a = parse(
        "--child", "env", "--child-out", "/tmp/o.json", "--scene-id", "grscenes:x", "--backend", "arena",
        "--spawn", "1.5", "2.0", "0.25", "0.01", "--shared-scene", "--num-envs", "1",
    )  # fmt: skip
    assert vs.check_args(a) is None
    assert a.spawn == [1.5, 2.0, 0.25, 0.01] and a.shared_scene is True and a.num_envs == 1
    assert parse().shared_scene is None  # backend default unless told


def test_child_command_goes_through_isaaclab_sh(tmp_path):
    a = parse("--isaaclab-sh", "/opt/il/isaaclab.sh", "--watchdog_s", "99")
    cmd = vs.child_command(a, "env", "grscenes:x_usd", "plain_usd", tmp_path / "o.json", ["--spawn", "1.0", "2.0", "0.1", "0.0"])
    assert cmd[:2] == ["/opt/il/isaaclab.sh", "-p"] and cmd[2].endswith("validate_scenes.py")
    assert cmd[3:5] == ["--child", "env"] and cmd[-5] == "--spawn"
    assert vs.check_args(vs.build_parser().parse_args(cmd[3:])) is None  # the child can parse what the parent sends


# ---- scene discovery ---------------------------------------------------------------------------------------
def test_grscenes_plan_is_smallest_first_and_filters_categories(scenes_dir):
    plans = vs.plan_scenes(parse())
    assert [(p.scene_id, p.category) for p in plans] == [
        ("grscenes:c_a_usd", "commercial"), ("grscenes:h_small_usd", "home"), ("grscenes:h_mid_usd", "home"),
        ("grscenes:h_big_usd", "home"), ("grscenes:c_b_usd", "commercial"),
    ]  # fmt: skip
    assert {p.backend for p in plans} == {"plain_usd"}
    only_home = vs.plan_scenes(parse("--categories", "home"))
    assert [p.scene_id for p in only_home] == ["grscenes:h_small_usd", "grscenes:h_mid_usd", "grscenes:h_big_usd"]


def test_explicit_scene_refs_accept_bare_ids(scenes_dir):
    plans = vs.plan_scenes(parse("--scene", "h_mid_usd", "--scene", "grscenes:c_b_usd"))
    assert [(p.scene_id, p.category) for p in plans] == [("grscenes:h_mid_usd", "home"), ("grscenes:c_b_usd", "commercial")]


def test_lightwheel_plan_is_a_stratified_subset_on_the_arena_backend():
    plans = vs.plan_scenes(parse("--source", "lightwheel", "--lightwheel-subset", "12"))
    assert len(plans) == len({p.scene_id for p in plans}) == 12
    assert plans[0].scene_id == "lightwheel_kitchen_one_wall_coastal"
    assert {p.backend for p in plans} == {"arena"} and {p.category for p in plans} == {"kitchen"}


def test_bad_scene_ref_is_exit_2(scenes_dir, capsys):
    assert vs.main(["--scene", "lightwheel:99:1", "--source", "lightwheel"]) == 2


# ---- dry run ------------------------------------------------------------------------------------------------
def test_dry_run_prints_plan_and_launches_nothing(scenes_dir, capsys, monkeypatch):
    monkeypatch.setattr(vs, "run_child", lambda *a, **k: pytest.fail("dry run launched a child"))
    assert vs.main(["--dry-run", "--limit", "2", "--arena-probe", "1", "--render"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["scenes_planned"] == 5 and plan["scenes_to_run"] == 4  # 2 per category
    assert plan["first_scenes"][:2] == ["grscenes:c_a_usd", "grscenes:h_small_usd"]
    assert any("render" in p for p in plan["phases"]) and any("arena probe" in p for p in plan["phases"])
    assert plan["example_sample_cmd"][3:5] == ["--child", "sample"]
    assert plan["thresholds"]["clear_height_m"] == 0.8


def test_dry_run_without_data_works_and_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CERBERUS_SCENES_DIR", str(tmp_path))
    assert vs.main(["--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["warnings"]
    assert vs.main([]) == 1  # a real run with nothing to do is an error


# ---- child processes ----------------------------------------------------------------------------------------
def test_run_child_reads_the_child_json(tmp_path):
    out = tmp_path / "o.json"
    code = f"import json; json.dump({{'ok': True, 'stage': 'done'}}, open({str(out)!r}, 'w'))"
    res = vs.run_child([sys.executable, "-c", code], 30, out, tmp_path / "o.log")
    assert res["ok"] and res["returncode"] == 0 and not res["timed_out"]


def test_run_child_kills_a_hung_process_group_and_keeps_partial_results(tmp_path):
    out = tmp_path / "o.json"
    pidfile = tmp_path / "grandchild.pid"
    code = (
        "import json, os, subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        f"json.dump({{'ok': False, 'stage': 'build_env'}}, open({str(out)!r}, 'w'))\n"
        "time.sleep(60)\n"
    )
    res = vs.run_child([sys.executable, "-c", code], 2, out, tmp_path / "o.log")
    assert res["timed_out"] and res["stage"] == "build_env" and res["wall_s"] < 30
    import os
    import time

    pid = int(pidfile.read_text())
    for _ in range(50):  # the whole process group is gone, not just the direct child
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("grandchild survived the timeout kill")


def test_run_child_without_a_result_file(tmp_path):
    res = vs.run_child([sys.executable, "-c", "raise SystemExit(3)"], 30, tmp_path / "none.json", tmp_path / "l.log")
    assert res["returncode"] == 3 and "ok" not in res


# ---- scene flow with a fake phase runner --------------------------------------------------------------------
class FakeRunner:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def __call__(self, phase, extra=None, *, backend=None, tag=""):
        self.calls.append((phase + tag, list(extra or []), backend))
        return dict(self.results[phase + tag])


PASS_RESULTS = {
    "sample": good_sample(),
    "env_4": good_env(),
    "env_16": good_env(steps_done=50, steps_requested=50),
    "render": {"ok": True, "frame_std": 30.0},
    "env_arena_probe": good_env(steps_done=50, steps_requested=50),
}
PLAN = vs.ScenePlan("grscenes:a_usd", "home", "plain_usd")


def run(args, runner, want_probe=False):
    return vs.run_scene(PLAN, parse(*args), TH, runner, want_probe=want_probe)


def test_passing_scene_runs_sample_then_env4_then_env16():
    runner = FakeRunner(PASS_RESULTS)
    row = run([], runner)
    assert row.status == "pass" and row.scene_id == "grscenes:a_usd" and row.category == "home"
    assert [c[0] for c in runner.calls] == ["sample", "env_4", "env_16"]
    env4_extra = runner.calls[1][1]
    assert env4_extra[:5] == ["--spawn", "1.0", "2.0", "0.5", "0.01"]  # the first, best spawn
    assert env4_extra[5:] == ["--num-envs", "4", "--steps", "200"]
    assert runner.calls[2][1][5:] == ["--num-envs", "16", "--steps", "50"]


def test_combined_env_phases_run_one_env_launch_at_the_long_step_count():
    results = {**PASS_RESULTS, "env_combined": good_env()}
    runner = FakeRunner(results)
    row = run(["--combined-env-phases"], runner)
    assert [c[0] for c in runner.calls] == ["sample", "env_combined"]  # no separate env_4
    assert runner.calls[1][1][5:] == ["--num-envs", "16", "--steps", "200"]  # scale envs, LONG step count
    assert row.status == "pass"
    assert row.metrics["env_phase"] == "combined" and row.metrics["build_s"] == 25.0


def test_combined_env_phases_skip_render_when_the_env_run_fails():
    results = {**PASS_RESULTS, "env_combined": good_env(tilt_deg_max=90.0)}
    runner = FakeRunner(results)
    row = run(["--combined-env-phases", "--render"], runner)
    assert [c[0] for c in runner.calls] == ["sample", "env_combined"]  # render never launched
    assert row.status == "fail" and any("tilt" in r for r in row.reasons)


def test_sample_failure_stops_before_any_env_is_built():
    runner = FakeRunner({"sample": good_sample(spawns=[])})
    row = run([], runner)
    assert row.status == "fail" and [c[0] for c in runner.calls] == ["sample"]


def test_env4_failure_skips_the_scale_run():
    runner = FakeRunner({**PASS_RESULTS, "env_4": {"timed_out": True, "wall_s": 480.0, "stage": "build_env"}})
    row = run([], runner)
    assert row.status == "fail" and "env4: timeout" in row.reasons[0]
    assert [c[0] for c in runner.calls] == ["sample", "env_4"]


def test_render_and_probe_only_when_requested():
    runner = FakeRunner(PASS_RESULTS)
    assert run([], runner).metrics.get("arena_probe") is None and "render" not in [c[0] for c in runner.calls]
    runner = FakeRunner(PASS_RESULTS)
    row = run(["--render"], runner, want_probe=True)
    assert [c[0] for c in runner.calls] == ["sample", "env_4", "env_16", "render", "env_arena_probe"]
    probe_call = runner.calls[-1]
    assert probe_call[2] == "arena" and "--shared-scene" in probe_call[1] and probe_call[1][-2:] != ["--num-envs", "4"]
    assert row.status == "pass" and row.metrics["arena_probe"]["status"] == "pass"


def test_blank_render_fails_only_with_render_flag():
    results = {**PASS_RESULTS, "render": {"ok": True, "frame_std": 0.0}}
    assert run(["--render"], FakeRunner(results)).status == "fail"
    assert run([], FakeRunner(results)).status == "pass"


# ---- render frame statistics (child side; the child itself needs a GPU) -------------------------------------
def test_frame_stats_measures_rgb_only_and_a_textured_frame_beats_the_threshold():
    np = pytest.importorskip("numpy")
    assert vs._frame_stats(np.zeros((4, 6, 3), dtype=np.uint8)) == {
        "frame_shape": [4, 6, 3], "frame_mean": 0.0, "frame_std": 0.0,
    }  # fmt: skip
    rgba = np.zeros((4, 6, 4), dtype=np.uint8)  # the rgb annotator hands back 4 channels
    rgba[..., 0] = 255
    stats = vs._frame_stats(rgba)
    assert stats["frame_shape"] == [4, 6, 4]
    assert stats["frame_mean"] == pytest.approx(255 / 3)  # the alpha channel is excluded
    assert stats["frame_std"] > TH.render_min_std  # i.e. the admission verdict would call it non-blank


def test_a_fully_transparent_frame_still_reads_blank():
    np = pytest.importorskip("numpy")
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[..., 3] = 255  # alpha alone must not look like content
    assert vs._frame_stats(rgba)["frame_std"] == 0.0


def test_failed_probe_is_recorded_but_scene_still_passes():
    results = {**PASS_RESULTS, "env_arena_probe": {"timed_out": True, "wall_s": 480.0, "stage": "build_env"}}
    row = run([], FakeRunner(results), want_probe=True)
    assert row.status == "pass" and row.metrics["arena_probe"]["status"] == "fail"


# ---- sweep: manifest after every scene, resume, limit ------------------------------------------------------
def fake_runner_factory(calls, bad=()):
    def make(args, plan, work_dir):
        def phase(name, extra=None, *, backend=None, tag=""):
            calls.append((plan.scene_id, name + tag))
            if plan.scene_id in bad and name == "sample":
                return good_sample(spawns=[])
            return dict(PASS_RESULTS[name + tag])

        return phase

    return make


def run_sweep(argv, calls, bad=()):
    args = parse(*argv)
    args.report_dir = str(args.report_dir)
    return vs.sweep(args, vs.plan_scenes(args), vs._abs(args.report_dir) / "report.json")


def test_sweep_writes_manifest_per_scene_and_resume_skips(scenes_dir, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(vs, "make_phase_runner", fake_runner_factory(calls, bad={"grscenes:h_mid_usd"}))
    report_dir = str(tmp_path / "reports")
    assert run_sweep(["--limit", "2", "--report-dir", report_dir], calls) == 0
    rows = {r.scene_id: r for r in admission.load_manifest("grscenes")}
    assert set(rows) == {"grscenes:c_a_usd", "grscenes:c_b_usd", "grscenes:h_small_usd", "grscenes:h_mid_usd"}
    assert rows["grscenes:h_mid_usd"].status == "fail" and rows["grscenes:c_a_usd"].status == "pass"
    assert admission.admitted_scene_ids("grscenes", "commercial") == ["grscenes:c_a_usd", "grscenes:c_b_usd"]
    assert admission.spawn_pose("grscenes", "grscenes:c_a_usd").x == 1.0
    report = json.loads((tmp_path / "reports" / "report.json").read_text())
    assert report["done"] and report["counts"] == {"pass": 3, "fail": 1, "skip": 0}
    n_first = len(calls)

    assert run_sweep(["--limit", "2", "--resume", "--report-dir", report_dir], calls) == 0  # next smallest, not the same
    assert {c[0] for c in calls[n_first:]} == {"grscenes:h_big_usd"}
    assert len(admission.load_manifest("grscenes")) == 5  # earlier rows kept


def test_sweep_without_resume_reruns_and_replaces_only_those_rows(scenes_dir, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(vs, "make_phase_runner", fake_runner_factory(calls))
    report_dir = str(tmp_path / "r")
    run_sweep(["--report-dir", report_dir], calls)
    calls.clear()
    run_sweep(["--scene", "h_mid_usd", "--report-dir", report_dir], calls)
    assert {c[0] for c in calls} == {"grscenes:h_mid_usd"}
    assert len(admission.load_manifest("grscenes")) == 5


def test_missing_scene_data_becomes_a_skip_row(scenes_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(vs, "make_phase_runner", fake_runner_factory([]))
    run_sweep(["--scene", "grscenes:ghost_usd", "--report-dir", str(tmp_path / "r")], [])
    (row,) = admission.load_manifest("grscenes")
    assert row.status == "skip" and "scene data missing" in row.reasons[0]


def test_probe_budget_counts_rows_already_probed(scenes_dir, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(vs, "make_phase_runner", fake_runner_factory(calls))
    run_sweep(["--limit", "1", "--arena-probe", "2", "--report-dir", str(tmp_path / "r")], calls)
    assert sum(1 for c in calls if c[1] == "env_arena_probe") == 2  # one per category: 2 scenes, budget 2
    calls.clear()
    run_sweep(["--limit", "1", "--resume", "--arena-probe", "2", "--report-dir", str(tmp_path / "r")], calls)
    assert not any(c[1] == "env_arena_probe" for c in calls)  # budget already spent


def test_max_runtime_stops_before_the_next_scene(scenes_dir, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(vs, "make_phase_runner", fake_runner_factory(calls))
    run_sweep(["--max-runtime-h", "0", "--report-dir", str(tmp_path / "r")], calls)
    assert len({c[0] for c in calls}) <= 1  # elapsed > 0 h after the first scene: nothing more is started


def test_tool_versions_are_read_from_the_image_file(tmp_path, monkeypatch):
    f = tmp_path / "v.json"
    f.write_text(json.dumps({"isaaclab_sha": "abc", "arena_sha": "def", "isaacsim_image": "6.1.0"}))
    monkeypatch.setattr(vs, "VERSIONS_FILE", f)
    assert vs.read_tool_versions()["arena_sha"] == "def"
    monkeypatch.setattr(vs, "VERSIONS_FILE", tmp_path / "missing.json")
    assert vs.read_tool_versions() == {}
