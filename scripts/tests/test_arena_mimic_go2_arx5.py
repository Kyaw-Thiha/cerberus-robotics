"""Pure-python tests for scripts/arena_mimic_go2_arx5.py (no Isaac, no GPU)."""

import json
import subprocess
import sys
from pathlib import Path

import arena_mimic_go2_arx5 as am
import pytest

from common.cerberus_worlds.arena_assets import go2_arx5_embodiment as emb

ROOT = Path(__file__).resolve().parents[2]


def parse(*argv):
    return am.build_parser().parse_args(list(argv))


def test_module_import_is_isaac_free():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
        "import arena_mimic_go2_arx5\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('isaaclab', 'isaaclab_arena', 'isaaclab_tasks', 'isaacsim', 'omni', 'pxr', 'torch')]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_script_never_imports_a_backend_directly():
    source = (ROOT / "scripts" / "arena_mimic_go2_arx5.py").read_text()
    assert "cerberus_worlds.backends" not in source
    assert "registry.get_backend(BACKEND)" in source and am.BACKEND == "arena"


# --- subcommand parsing -------------------------------------------------------------------------------------
def test_subcommand_required_and_known():
    with pytest.raises(SystemExit):
        parse()
    with pytest.raises(SystemExit):
        parse("train")
    assert am.COMMANDS == ("smoke", "record", "annotate", "generate")


@pytest.mark.parametrize("command", am.COMMANDS)
def test_every_subcommand_has_the_common_flags(command):
    args = parse(command)
    assert args.command == command
    assert args.num_envs == am.DEFAULT_NUM_ENVS[command]
    assert args.report == am.DEFAULT_REPORTS[command]
    assert args.headless is False and args.dry_run is False and args.watchdog_s == 900.0
    args = parse(command, "--num_envs", "3", "--report", "r.json", "--headless", "--watchdog_s", "5", "--dry-run")
    assert (args.num_envs, args.report, args.headless, args.watchdog_s, args.dry_run) == (3, "r.json", True, 5.0, True)
    assert parse(command, "--dry_run").dry_run is True
    # kitchen flags are shared by all stages
    assert parse(command, "--robot_xy", "2", "-1").robot_xy == [2.0, -1.0]
    assert parse(command).kitchen_scene == emb.DEFAULT_KITCHEN_REF


def test_stage_specific_flags_and_defaults():
    assert parse("smoke").steps == 30 and parse("smoke").env_kind == "mimic"
    r = parse(
        "record", "--num_demos", "3", "--output", "x.hdf5", "--grasp_z_offset", "0.01", "--grasp_yaw_deg", "37.5"
    )
    assert (r.num_demos, r.output, r.grasp_z_offset, r.grasp_yaw_deg) == (3, "x.hdf5", 0.01, 37.5)
    assert parse("record").grasp_yaw_deg == pytest.approx(90.0)
    a = parse("annotate")
    assert (a.input_file, a.output_file, a.engine) == (am.DEFAULT_RECORDED, am.DEFAULT_ANNOTATED, "inprocess")
    g = parse("generate")
    assert (g.input_file, g.output_file, g.generation_num_trials) == (am.DEFAULT_ANNOTATED, am.DEFAULT_GENERATED, 10)
    assert g.num_envs == 4 and g.arena_root == am.DEFAULT_ARENA_ROOT


def test_object_yaw_extraction_and_grasp_quaternion_order():
    calls = []
    extracted_yaw = object()

    class FakeMathUtils:
        @staticmethod
        def euler_xyz_from_quat(quaternion):
            calls.append(("euler_xyz_from_quat", quaternion))
            return object(), object(), extracted_yaw

        @staticmethod
        def quat_mul(left, right):
            calls.append(("quat_mul", left, right))
            return ("mul", left, right)

    object_quat, q_yaw, q_pitch, eef = object(), object(), object(), object()
    yaw = am._yaw_from_quaternion(FakeMathUtils, object_quat)
    result = am._compose_grasp_orientation(FakeMathUtils, q_yaw, q_pitch, eef)

    assert yaw is extracted_yaw
    assert calls == [
        ("euler_xyz_from_quat", object_quat),
        ("quat_mul", q_yaw, q_pitch),
        ("quat_mul", ("mul", q_yaw, q_pitch), eef),
    ]
    assert result == ("mul", ("mul", q_yaw, q_pitch), eef)

def test_object_yaw_extraction_converts_lab_proxy_array():
    tensor = object()
    proxy = type("ProxyArray", (), {"torch": tensor})()
    seen = []

    class FakeMathUtils:
        @staticmethod
        def euler_xyz_from_quat(quaternion):
            seen.append(quaternion)
            return None, None, "yaw"

    assert am._yaw_from_quaternion(FakeMathUtils, proxy) == "yaw"
    assert seen == [tensor]




def test_validate_args():
    assert am.validate_args(parse("smoke")) == []
    assert any("--grasp_yaw_deg must be finite" in issue for issue in am.validate_args(parse("record", "--grasp_yaw_deg", "nan")))
    assert am.validate_args(parse("record", "--num_envs", "2"))  # single-env recorder
    assert am.validate_args(parse("annotate", "--num_envs", "2"))
    assert am.validate_args(parse("generate", "--num_envs", "8")) == []
    assert am.validate_args(parse("smoke", "--num_envs", "0"))
    assert am.validate_args(parse("smoke", "--steps", "0"))
    assert am.validate_args(parse("record", "--num_demos", "0"))
    assert am.validate_args(parse("record", "--output", "demos.txt"))
    assert am.validate_args(parse("record", "--hover_height", "0"))
    assert am.validate_args(parse("generate", "--input_file", "in.txt"))
    assert am.validate_args(parse("generate", "--seed", "10", "--generation_num_trials", "10"))  # Lab token check
    assert am.validate_args(parse("generate", "--timeout_s", "0"))


def test_attempt_budget_and_waypoint_config():
    assert am.attempt_budget(parse("record", "--num_demos", "4")) == 17
    assert am.attempt_budget(parse("record", "--num_demos", "4", "--max_attempts", "6")) == 6
    cfg = am.pick_place_config(parse("record", "--hover_height", "0.12", "--object_height_m", "0.05"))
    assert cfg.hover_height == 0.12 and cfg.cube_size == 0.05 and cfg.ik_scale == am.layout.IK_ACTION_SCALE


# --- dry-run plans -------------------------------------------------------------------------------------------
def run_dry(capsys, *argv):
    rc = am.main([*argv, "--dry-run"])
    captured = capsys.readouterr()
    return rc, (json.loads(captured.out) if captured.out.strip() else None), captured.err


def test_smoke_dry_run_resolves_the_plan(capsys):
    rc, plan, err = run_dry(capsys, "smoke")
    assert rc == 0, err
    assert plan["command"] == "smoke" and plan["backend"] == "arena" and plan["backend_class"] == "ArenaBackend"
    assert plan["scene_id"] == "lightwheel_kitchen_one_wall_coastal"
    assert plan["scene"]["kind"] == "lightwheel"
    e = plan["embodiment"]
    assert e["name"] == "go2_arx5" and e["fixed_base"] is True and e["leg_action_term"] is None
    assert e["action_dim"] == 7
    assert e["arm_action"]["body_name"] == "eef_link" and e["arm_action"]["scale"] == 0.5
    assert e["arm_action"]["joint_names"] == [f"joint{i}" for i in range(1, 7)]
    assert e["gripper_action"]["joint_names"] == ["joint7", "joint8"]
    assert "eef_pos" in e["observation_terms"] and "eef_quat" in e["observation_terms"]
    assert e["ee_frame"]["name"] == "ee_frame"
    assert len(e["joint_groups"]["legs"]) == 12
    assert plan["environment"]["name"] == emb.ENV_NAME and plan["environment"]["options"]["pick_object"] == "sweet_potato"
    assert plan["environment"]["geometry"]["robot_yaw_rad"] == 0.0
    assert plan["steps"] == 30 and plan["env_kind"] == "mimic" and plan["problems"] == []


def test_dry_run_accepts_scene_refs_and_flags(capsys):
    rc, plan, _ = run_dry(
        capsys, "smoke", "--kitchen_scene", "lightwheel:3:5", "--num_envs", "4", "--env_kind", "rl", "--pick_object", "jug"
    )
    assert rc == 0
    assert plan["scene_id"] == "lightwheel_kitchen_l_shaped_modern1"
    assert plan["num_envs"] == 4 and plan["env_kind"] == "rl" and plan["environment"]["options"]["pick_object"] == "jug"
    rc, plan, _ = run_dry(capsys, "smoke", "--kitchen_scene", "lightwheel_kitchen_galley_rustic")
    assert rc == 0 and plan["scene_id"] == "lightwheel_kitchen_galley_rustic"


def test_dry_run_bad_scene_ref_is_exit_2(capsys):
    rc, plan, err = run_dry(capsys, "smoke", "--kitchen_scene", "lightwheel:11:1")
    assert rc == 2 and plan is None and "Bad Lightwheel ref" in err


def test_dry_run_non_lightwheel_scene_is_a_problem(capsys):
    rc, plan, err = run_dry(capsys, "smoke", "--kitchen_scene", "stock")
    assert rc == 1 and plan["problems"] and "Lightwheel" in err


def test_record_dry_run_plan(capsys, tmp_path):
    rc, plan, err = run_dry(capsys, "record", "--num_demos", "4", "--output", str(tmp_path / "d.hdf5"))
    assert rc == 0, err
    assert plan["num_demos"] == 4 and plan["attempt_budget"] == 17
    assert plan["waypoints"]["ik_scale"] == 0.5 and plan["waypoints"]["cube_size"] == 0.04
    assert plan["grasp_yaw_offset_deg"] == pytest.approx(90.0)
    assert plan["grasp_yaw_mode"] == "object_yaw_plus_offset"
    assert plan["output"].endswith("d.hdf5")


def test_record_dry_run_refuses_to_overwrite(capsys, tmp_path):
    out = tmp_path / "d.hdf5"
    out.write_text("x")
    rc, plan, err = run_dry(capsys, "record", "--output", str(out))
    assert rc == 1 and "exists" in err
    rc, _, _ = run_dry(capsys, "record", "--output", str(out), "--overwrite")
    assert rc == 0


def test_annotate_dry_run_default_engine_is_inprocess(capsys):
    rc, plan, err = run_dry(capsys, "annotate")
    assert rc == 0, err
    assert plan["engine"] == "inprocess" and "lab_command" not in plan
    assert plan["input_file"] == am.DEFAULT_RECORDED


def test_annotate_dry_run_lab_engine_command(capsys):
    rc, plan, err = run_dry(capsys, "annotate", "--engine", "lab")
    assert rc == 0, err
    cmd = plan["lab_command"]
    assert cmd[1].endswith("isaaclab_mimic/annotate_demos.py")
    assert "--auto" in cmd and "--mimic" in cmd
    assert cmd[cmd.index("--task") + 1] == emb.ENV_NAME
    assert cmd[cmd.index("--external_callback") + 1] == am.INTEROP_CALLBACK
    assert cmd[cmd.index("--external_environment_class_path") + 1] == emb.ENV_FACTORY_PATH
    assert "args_cli.headless" in err  # known Lab-side problem is surfaced


def test_generate_dry_run_command(capsys):
    rc, plan, err = run_dry(
        capsys, "generate", "--generation_num_trials", "25", "--num_envs", "6", "--robot_xy", "2.0", "-2.0"
    )
    assert rc == 0, err
    cmd = plan["lab_command"]
    assert cmd[1].endswith("submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py")
    assert cmd[cmd.index("--generation_num_trials") + 1] == "25" and cmd[cmd.index("--num_envs") + 1] == "6"
    assert "--mimic" in cmd and "--no_solve_relations" in cmd and "--auto" not in cmd
    # the kitchen flags travel to the interop callback, same values as the other stages
    i = cmd.index("--robot_xy")
    assert cmd[i + 1 : i + 3] == ["2.0", "-2.0"]
    assert cmd[cmd.index("--kitchen_scene") + 1] == "lightwheel:1:1"
    assert plan["lab_command_line"].startswith(cmd[0])


def test_lab_env_makes_repo_importable_and_headless(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    env = am.lab_env()
    assert env["HEADLESS"] == "1"
    assert env["PYTHONPATH"].split(":")[0] == str(am.Path(am.__file__).resolve().parents[1])
    assert "/somewhere" in env["PYTHONPATH"]
    assert "HEADLESS" not in {k: v for k, v in am.lab_env(headless=False).items() if k not in __import__("os").environ}


# --- evaluation functions --------------------------------------------------------------------------------------
def _good_smoke() -> dict:
    return {
        "env_kind": "mimic",
        "num_envs": 1,
        "pick_name": "sweet_potato",
        "action_dim": 7,
        "action_terms": ["arm_action", "gripper_action"],
        "policy_obs_keys": list(emb.OBSERVATION_TERMS),
        "subtask_obs_keys": ["grasp_1"],
        "scene_keys": ["robot", "ee_frame", "sweet_potato", emb.PLATE_NAME],
        "is_fixed_base": True,
        "finite": True,
        "base_z_drift": 0.0,
        "arm_drift_rad": 0.001,
        "pick_dz_to_support": 0.03,
        "plate_dz_to_support": 0.006,
        "mimic_eef_pose_shape": [1, 4, 4],
        "mimic_object_pose_keys": ["sweet_potato", emb.PLATE_NAME],
        "mimic_signal_keys": ["grasp_1"],
    }


def test_evaluate_smoke_pass_and_each_failure():
    assert am.evaluate_smoke(_good_smoke()) == []
    cases = {
        "action_dim": 9,
        "action_terms": ["arm_action", "legs", "gripper_action"],
        "policy_obs_keys": ["actions"],
        "subtask_obs_keys": [],
        "scene_keys": ["robot"],
        "is_fixed_base": False,
        "finite": False,
        "base_z_drift": 0.1,
        "arm_drift_rad": 0.5,
        "pick_dz_to_support": -1.0,
        "mimic_eef_pose_shape": [1, 4],
        "mimic_object_pose_keys": [],
        "mimic_signal_keys": [],
    }
    for key, value in cases.items():
        m = _good_smoke()
        m[key] = value
        assert am.evaluate_smoke(m), key
    rl = _good_smoke()
    rl.update(env_kind="rl", mimic_eef_pose_shape=None, mimic_object_pose_keys=None, mimic_signal_keys=None)
    assert am.evaluate_smoke(rl) == []


def test_evaluate_record():
    ok = {"num_demos": 2, "success": [True, True]}
    assert am.evaluate_record(2, 2, ok, None) == []
    assert am.evaluate_record(3, 2, ok, None)
    assert am.evaluate_record(2, 2, None, None)
    assert am.evaluate_record(2, 2, {"error": "x"}, None)
    assert am.evaluate_record(2, 2, {"num_demos": 1, "success": [True]}, None)
    assert am.evaluate_record(2, 2, {"num_demos": 2, "success": [True, False]}, None)
    assert am.evaluate_record(2, 2, ok, "boom")


def test_failure_diagnostics_include_terminal_pose_errors_and_grasp_signal():
    diagnostic = am._failure_diagnostics(
        obs={
            "policy": {"eef_pos": [[0.30, 0.10, 0.33]], "gripper_pos": [[0.02, 0.03]]},
            "subtask_terms": {"grasp_1": [[True]]},
        },
        target_pos=(0.32, 0.10, 0.33),
        rot_error=(0.0, 0.1, 0.0),
        pick_pose={"position_m": [0.32, 0.1, 0.33], "root_quaternion_wxyz": [1, 0, 0, 0]},
        phase="descend_pick",
        joint_state=[{"name": "joint", "position_rad": 0.5, "soft_limits_rad": [-1, 1]}],
    )
    assert diagnostic["phase"] == "descend_pick"
    assert diagnostic["eef_position_m"] == [0.30, 0.10, 0.33]
    assert diagnostic["target_eef_position_m"] == [0.32, 0.10, 0.33]
    assert diagnostic["position_error_vector_m"] == pytest.approx([0.02, 0.0, 0.0])
    assert diagnostic["position_error_m"] == pytest.approx(0.02)
    assert diagnostic["finger_positions_m"] == [0.02, 0.03]
    assert diagnostic["rotation_error_vector_rad"] == [0.0, 0.1, 0.0]
    assert diagnostic["grasp_1_signal"] is True
    assert diagnostic["pick_object_root_pose"]["position_m"] == [0.32, 0.1, 0.33]
    assert diagnostic["arm_joint_state"][0]["position_rad"] == 0.5
    assert diagnostic["arm_joint_state"][0]["soft_limits_rad"] == [-1.0, 1.0]


def test_evaluate_annotate():
    assert am.evaluate_annotate(3, 3, {"num_demos": 3}) == []
    assert am.evaluate_annotate(0, 0, None)
    assert am.evaluate_annotate(3, 2, {"num_demos": 2})
    assert am.evaluate_annotate(3, 3, {"num_demos": 2})


def test_evaluate_generate():
    ok = {"num_demos": 5, "success": [True] * 5}
    assert am.evaluate_generate(0, False, ok, 5, False) == []
    assert am.evaluate_generate(1, False, ok, 5, False)
    assert am.evaluate_generate(None, True, ok, 5, False)  # killed on timeout
    assert am.evaluate_generate(None, True, ok, 5, True) == []  # explicitly accepted
    assert am.evaluate_generate(0, False, ok, 6, False)
    assert am.evaluate_generate(0, False, None, 1, False)
    assert am.evaluate_generate(0, False, {"num_demos": 1, "success": [False]}, 1, False)


# --- main() flow with fakes: report first, then hard exit with the right code ----------------------------------
@pytest.fixture
def fake_main(monkeypatch, tmp_path):
    state = {"exit": None, "launched": 0, "report_at_exit": None}

    def finish(code):
        state["exit"] = code
        state["report_at_exit"] = report_path.exists()

    report_path = tmp_path / "r.json"
    monkeypatch.setattr(am, "_finish", finish)
    monkeypatch.setattr(am, "_launch_app", lambda args: state.__setitem__("launched", state["launched"] + 1))
    monkeypatch.setattr(am, "_purge_leaked_isaaclab_assets", lambda: None)
    return state, report_path


@pytest.mark.parametrize("verdict, expected", [(True, 0), (False, 1)])
def test_main_writes_report_before_exit(monkeypatch, fake_main, verdict, expected):
    state, report_path = fake_main

    def runner(args, plan, report, path):
        report["passed"], report["failure_reasons"] = verdict, ([] if verdict else ["nope"])

    monkeypatch.setitem(am.RUNNERS, "smoke", runner)
    rc = am.main(["smoke", "--report", str(report_path), "--watchdog_s", "0"])
    assert state["exit"] == expected and rc == expected
    assert state["report_at_exit"] is True and state["launched"] == 1
    data = json.loads(report_path.read_text())
    assert data["passed"] is verdict and data["command"] == "smoke" and data["plan"]["scene_id"]


def test_main_runner_exception_is_a_failed_report(monkeypatch, fake_main):
    state, report_path = fake_main

    def runner(args, plan, report, path):
        report["progress"]["stage"] = "build"
        raise RuntimeError("kaboom")

    monkeypatch.setitem(am.RUNNERS, "record", runner)
    am.main(["record", "--report", str(report_path), "--output", str(report_path.parent / "o.hdf5"), "--watchdog_s", "0"])
    assert state["exit"] == 1
    data = json.loads(report_path.read_text())
    assert data["passed"] is False and "kaboom" in json.dumps(data["failure_reasons"]) and "build" in json.dumps(
        data["failure_reasons"]
    )


def test_generate_does_not_launch_the_app(monkeypatch, fake_main):
    state, report_path = fake_main
    monkeypatch.setitem(
        am.RUNNERS, "generate", lambda a, p, r, path: r.update(passed=True, failure_reasons=[])
    )
    am.main(["generate", "--report", str(report_path), "--output_file", str(report_path.parent / "g.hdf5"), "--watchdog_s", "0"])
    assert state["launched"] == 0 and state["exit"] == 0


def test_lab_subprocess_failure_and_timeout(monkeypatch, tmp_path):
    args = parse(
        "generate", "--report", str(tmp_path / "rep.json"), "--output_file", str(tmp_path / "g.hdf5"), "--timeout_s", "30"
    )
    plan = {"lab_command": [sys.executable, "-c", "raise SystemExit(3)"], "lab_command_line": "x"}
    report = am._new_report(args, plan)
    am._run_lab_subprocess(args, plan, report, tmp_path / "rep.json", "generate")
    assert report["passed"] is False and report["metrics"]["returncode"] == 3
    assert any("exited with code 3" in r for r in report["failure_reasons"])
    assert (tmp_path / "rep.log").exists()

    args.timeout_s = 0.5
    plan = {"lab_command": [sys.executable, "-c", "import time; time.sleep(30)"], "lab_command_line": "x"}
    report = am._new_report(args, plan)
    am._run_lab_subprocess(args, plan, report, tmp_path / "rep.json", "generate")
    assert report["metrics"]["timed_out"] is True and report["passed"] is False
    assert any("--timeout_s" in r for r in report["failure_reasons"])


def test_main_launch_failure_still_leaves_a_failed_report(monkeypatch, fake_main):
    state, report_path = fake_main

    def boom(args):
        raise RuntimeError("no GPU")

    monkeypatch.setattr(am, "_launch_app", boom)
    am.main(["smoke", "--report", str(report_path), "--watchdog_s", "0"])
    assert state["exit"] == 1 and state["report_at_exit"] is True
    data = json.loads(report_path.read_text())
    assert data["passed"] is False and "no GPU" in json.dumps(data["failure_reasons"])
