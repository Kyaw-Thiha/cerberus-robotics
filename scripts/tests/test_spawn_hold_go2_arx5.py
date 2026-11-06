import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest
import spawn_hold_go2_arx5 as sh

T = sh.HoldThresholds()


def good():
    return {
        "all_finite": True,
        "base_height_end_min": 0.30,
        "base_height_end_max": 0.34,
        "tilt_deg_max": 2.0,
        "arm_joint_err_max": 0.03,
        "leg_joint_err_max": 0.60,
        "settle_drift_m": 0.001,
        "joint_names_ok": True,
        "eef_present": True,
        "arm_mass_kg": 3.5,
    }


def good_gripper():
    return {
        **good(),
        "num_dof": 20,
        "gripper_tested": True,
        "gripper_closed_err_max": 0.001,
        "gripper_open_err_max": 0.001,
    }


def test_pass():
    ok, reasons = sh.evaluate_hold(good(), T)
    assert ok and reasons == []


def test_pass_with_gripper_and_dof():
    ok, reasons = sh.evaluate_hold(good_gripper(), T)
    assert ok and reasons == []


@pytest.mark.parametrize(
    "patch, needle",
    [
        ({"num_dof": 18}, "20"),
        ({"gripper_closed_err_max": 0.02}, "did not close"),
        ({"gripper_open_err_max": 0.02}, "did not open"),
        ({"gripper_open_err_max": float("nan")}, "gripper"),
        ({"gripper_closed_err_max": None}, "gripper"),
    ],
)
def test_gripper_and_dof_failure_branches(patch, needle):
    ok, reasons = sh.evaluate_hold({**good_gripper(), **patch}, T)
    assert not ok
    assert len(reasons) == 1 and needle in reasons[0]


def test_gripper_checks_skipped_when_not_tested():
    m = {**good(), "gripper_tested": False, "gripper_closed_err_max": 1.0}
    ok, reasons = sh.evaluate_hold(m, T)
    assert ok and reasons == []


@pytest.mark.parametrize(
    "patch, needle",
    [
        ({"all_finite": False}, "finite"),
        ({"base_height_end_max": float("nan")}, "finite"),
        ({"base_height_end_min": 0.05}, "too low"),
        ({"base_height_end_max": 0.9}, "too high"),
        ({"tilt_deg_max": 60.0}, "tilt"),
        ({"arm_joint_err_max": 0.5}, "arm joint error"),
        ({"leg_joint_err_max": 1.2}, "leg joint error"),
        ({"settle_drift_m": 0.2}, "settling"),
        ({"joint_names_ok": False}, "joint-name"),
        ({"eef_present": False}, "EEF"),
        ({"arm_mass_kg": 2.0}, "arm mass"),
        ({"arm_mass_kg": 5.0}, "arm mass"),
    ],
)
def test_each_failure_branch(patch, needle):
    m = {**good(), **patch}
    ok, reasons = sh.evaluate_hold(m, T)
    assert not ok
    assert len(reasons) == 1 and needle in reasons[0]


def test_thresholds_defaults_and_frozen():
    assert (T.min_base_height, T.max_base_height) == (0.15, 0.45)
    assert T.max_tilt_deg == 25.0
    assert T.max_arm_joint_err_rad == 0.10
    assert T.max_leg_joint_err_rad == 0.80
    assert T.max_settle_drift_m == 0.02
    assert T.arm_mass_range == (3.2, 3.8)
    assert T.max_gripper_closed_err_m == 0.005
    assert T.max_gripper_open_err_m == 0.003
    with pytest.raises(FrozenInstanceError):
        T.max_tilt_deg = 1.0
    assert replace(T, max_tilt_deg=1.0).max_tilt_deg == 1.0


def test_arg_defaults():
    a = sh.build_parser().parse_args([])
    assert a.num_envs == 16 and a.duration_s == 10.0 and a.usd is None
    assert a.report == "data/reports/go2_arx5_hold.json"
    assert sh.thresholds_from_args(a) == T
    assert a.fix_base is False and a.no_gripper_test is False and a.gripper_settle_s == 1.0
    a = sh.build_parser().parse_args(["--fix_base", "--no_gripper_test", "--gripper_settle_s", "0.5"])
    assert a.fix_base is True and a.no_gripper_test is True and a.gripper_settle_s == 0.5
    a = sh.build_parser().parse_args(["--max_gripper_closed_err_m", "0.01"])
    assert sh.thresholds_from_args(a).max_gripper_closed_err_m == 0.01
    a = sh.build_parser().parse_args(["--arm_mass_range", "1", "2"])
    assert sh.thresholds_from_args(a).arm_mass_range == (1.0, 2.0)


def test_expected_dof_and_gripper_constants():
    assert sh.EXPECTED_DOF == 20
    assert sh.GRIPPER_JOINT_NAMES == ("joint7", "joint8")
    assert (sh.GRIPPER_OPEN_M, sh.GRIPPER_CLOSED_M) == (0.044, 0.0)


def test_bad_args_exit_2():
    with pytest.raises(SystemExit) as e:
        sh.build_parser().parse_args(["--num_envs", "abc"])
    assert e.value.code == 2


def test_import_does_not_pull_isaac():
    code = (
        "import sys; sys.path.insert(0, 'scripts'); import spawn_hold_go2_arx5; "
        "bad=[m for m in sys.modules if m.split('.')[0] in ('isaaclab','omni','isaacsim')]; "
        "sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


class _FakeApp:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


@pytest.mark.parametrize("code", [1, 2, 3])
def test_finish_nonzero_uses_os_exit_without_close(monkeypatch, code):
    app = _FakeApp()
    seen = []

    def fake_exit(c):
        seen.append(c)
        raise RuntimeError("exited")

    monkeypatch.setattr(sh.os, "_exit", fake_exit)
    with pytest.raises(RuntimeError, match="exited"):
        sh._finish(app, code)
    assert seen == [code]
    assert app.closed == 0


def test_finish_zero_hard_exits_without_close(monkeypatch):
    app = _FakeApp()
    seen = []
    monkeypatch.setattr(sh.os, "_exit", seen.append)
    sh._finish(app, 0)
    assert seen == [0]
    assert app.closed == 0


def test_watchdog_flag_default_and_disable():
    assert sh.build_parser().parse_args([]).watchdog_s == 420.0
    assert sh.build_parser().parse_args(["--watchdog_s", "0"]).watchdog_s == 0.0
