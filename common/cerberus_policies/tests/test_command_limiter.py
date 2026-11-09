"""Tests for the B1 command slew-rate limiter (pure Python, no torch/ROS/Isaac)."""

import math
import subprocess
import sys
from pathlib import Path

import pytest

from common.cerberus_policies.command_limiter import (
    DEFAULT_CONFIG_PATH,
    CommandLimiterConfigError,
    CommandRateLimiter,
)

CMD = {
    "vx": 0.0,
    "vy": 0.0,
    "wz": 0.0,
    "body_pitch": 0.0,
    "body_height": 0.0,
    "has_body_pitch": False,
    "has_body_height": False,
}


def limiter(**overrides) -> CommandRateLimiter:
    rate = {"vx": 1.0, "vy": 1.0, "wz": 2.0}
    rate.update(overrides.pop("rate", {}))
    clamp = {"vx": 1.0, "vy": 1.0, "wz": 1.0}
    clamp.update(overrides.pop("clamp", {}))
    return CommandRateLimiter(max_rate=rate, max_abs=clamp)


def cmd(**overrides) -> dict:
    return {**CMD, **overrides}


def test_step_input_ramps_at_the_configured_rate():
    lim = limiter()
    assert lim.apply(cmd(), dt=0.1)["vx"] == pytest.approx(0.0)  # adopts 0
    # a step to 1.0 m/s at 1 m/s^2 moves 0.1 m/s per 0.1 s
    assert lim.apply(cmd(vx=1.0), dt=0.1)["vx"] == pytest.approx(0.1)
    assert lim.apply(cmd(vx=1.0), dt=0.1)["vx"] == pytest.approx(0.2)
    for _ in range(20):
        last = lim.apply(cmd(vx=1.0), dt=0.1)
    assert last["vx"] == pytest.approx(1.0)


def test_full_sign_reversal_takes_rate_time():
    lim = limiter()
    lim.apply(cmd(vx=1.0), dt=1.0)  # adopt +1.0
    # +1.0 -> -1.0 is 2.0 m/s at 1 m/s^2 = 2.0 s
    assert lim.apply(cmd(vx=-1.0), dt=1.0)["vx"] == pytest.approx(0.0)
    assert lim.apply(cmd(vx=-1.0), dt=1.0)["vx"] == pytest.approx(-1.0)
    # and one 2.0 s step reaches the same value in a single apply
    lim.reset()
    lim.apply(cmd(vx=1.0), dt=1.0)
    assert lim.apply(cmd(vx=-1.0), dt=2.0)["vx"] == pytest.approx(-1.0)


def test_per_axis_independence_and_units():
    lim = limiter(rate={"vx": 0.5, "vy": 2.0, "wz": 4.0})
    lim.apply(cmd(), dt=1.0)
    out = lim.apply(cmd(vx=1.0, vy=1.0, wz=1.0), dt=1.0)
    assert out["vx"] == pytest.approx(0.5)  # m/s^2
    assert out["vy"] == pytest.approx(1.0)  # already at its clamp
    assert out["wz"] == pytest.approx(1.0)  # rad/s^2 reaches the clamp
    lim.reset()
    lim.apply(cmd(), dt=1.0)
    assert lim.apply(cmd(wz=1.0), dt=0.1)["wz"] == pytest.approx(0.4)


def test_clamp_is_per_axis_and_applies_before_the_rate_limit():
    lim = limiter(clamp={"vx": 0.3, "wz": 0.2}, rate={"vx": 10.0, "wz": 10.0})
    lim.apply(cmd(), dt=1.0)
    out = lim.apply(cmd(vx=5.0, wz=-5.0), dt=1.0)
    assert out["vx"] == pytest.approx(0.3)
    assert out["wz"] == pytest.approx(-0.2)


def test_small_steps_equal_one_big_step_and_bad_dt_raises():
    a, b = limiter(), limiter()
    a.apply(cmd(), dt=0.1)
    b.apply(cmd(), dt=0.1)
    for _ in range(10):
        a.apply(cmd(vx=1.0), dt=0.1)
    b.apply(cmd(vx=1.0), dt=1.0)
    assert a.apply(cmd(vx=1.0), dt=0.1)["vx"] == pytest.approx(b.apply(cmd(vx=1.0), dt=0.1)["vx"], abs=1e-9)
    for bad in (0.0, -0.1, float("nan"), float("inf"), True, "0.1"):
        with pytest.raises(ValueError, match="dt must be"):
            limiter().apply(cmd(), dt=bad)


def test_reset_forgets_state_and_next_apply_adopts():
    lim = limiter()
    lim.apply(cmd(vx=1.0), dt=1.0)
    assert lim.apply(cmd(vx=1.0), dt=1.0)["vx"] == pytest.approx(1.0)
    lim.reset()
    # no previous command to slew from: the clamped request is adopted, not ramped from zero
    assert lim.apply(cmd(vx=0.5), dt=0.001)["vx"] == pytest.approx(0.5)
    lim.reset()
    assert lim.apply(cmd(vx=-1.0), dt=0.001)["vx"] == pytest.approx(-1.0)


def test_optional_body_fields_pass_through_clamped_but_never_rate_limited():
    lim = limiter(clamp={"body_pitch": 0.2, "body_height": 0.15})
    out = lim.apply(cmd(vx=0.0, body_pitch=3.0, body_height=-3.0, has_body_pitch=True, has_body_height=True), dt=0.001)
    assert out["body_pitch"] == pytest.approx(0.2)
    assert out["body_height"] == pytest.approx(-0.15)
    assert out["has_body_pitch"] is True and out["has_body_height"] is True
    # without a clamp entry the body fields are untouched
    out = limiter().apply(cmd(body_pitch=3.0, body_height=-3.0), dt=0.001)
    assert out["body_pitch"] == pytest.approx(3.0) and out["body_height"] == pytest.approx(-3.0)


def test_unknown_fields_are_copied_through_and_input_is_not_mutated():
    lim = limiter()
    source = cmd(vx=0.4, has_body_height=True, extra="kept")
    out = lim.apply(source, dt=0.1)
    assert out["extra"] == "kept" and out["has_body_height"] is True
    assert source["vx"] == pytest.approx(0.4)  # input dict untouched
    assert out is not source


def test_idempotent_when_already_within_limits():
    lim = limiter()
    lim.apply(cmd(vx=0.2, vy=-0.3, wz=0.4), dt=0.1)
    first = lim.apply(cmd(vx=0.2, vy=-0.3, wz=0.4), dt=0.1)
    second = lim.apply(cmd(vx=0.2, vy=-0.3, wz=0.4), dt=0.1)
    assert first == second


def test_config_validation_errors():
    with pytest.raises(CommandLimiterConfigError, match="missing axes"):
        CommandRateLimiter(max_rate={"vx": 1.0}, max_abs={"vx": 1.0, "vy": 1.0, "wz": 1.0})
    with pytest.raises(CommandLimiterConfigError, match="unknown axes"):
        CommandRateLimiter(max_rate={"vx": 1.0, "vy": 1.0, "wz": 1.0, "roll": 1.0}, max_abs={"vx": 1.0, "vy": 1.0, "wz": 1.0})
    with pytest.raises(CommandLimiterConfigError, match="positive finite"):
        CommandRateLimiter(max_rate={"vx": 0.0, "vy": 1.0, "wz": 1.0}, max_abs={"vx": 1.0, "vy": 1.0, "wz": 1.0})
    with pytest.raises(CommandLimiterConfigError, match="positive finite"):
        CommandRateLimiter(max_rate={"vx": 1.0, "vy": 1.0, "wz": 1.0}, max_abs={"vx": 1.0, "vy": float("nan"), "wz": 1.0})
    with pytest.raises(CommandLimiterConfigError, match="max_rate.*max_abs"):
        CommandRateLimiter.from_config({"max_rate": {}})


def test_shipped_config_loads_and_caps_at_the_trained_command_ranges():
    lim = CommandRateLimiter.from_yaml(DEFAULT_CONFIG_PATH)
    assert lim.max_rate == {"vx": 1.0, "vy": 1.0, "wz": 2.0}
    # training command ranges: +-1.0 m/s on x/y, +-1.0 rad/s on yaw
    assert lim.max_abs == {"vx": 1.0, "vy": 1.0, "wz": 1.0}
    out = lim.apply(cmd(vx=9.0, vy=-9.0, wz=9.0), dt=10.0)
    assert (out["vx"], out["vy"], out["wz"]) == (1.0, -1.0, 1.0)


def test_imports_with_no_heavy_deps():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pixi.toml").exists())
    script = (
        "import sys\n"
        "from common.cerberus_policies.command_limiter import CommandRateLimiter\n"
        "bad = [m for m in ('torch', 'numpy', 'rclpy', 'isaaclab', 'isaacsim', 'omni', 'cerberus_msgs') if m in sys.modules]\n"
        "print('BAD:' + ','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], cwd=root, env={"PYTHONPATH": str(root), "PATH": ""}, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("BAD:"), out.stdout


def test_config_file_is_flagged_as_placeholder():
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").lower()
    assert "placeholder" in text and "untested" in text
    # the numbers themselves must remain honest guesses, not a claim of a fix
    assert "not tuned" in text
    assert math.isfinite(sum(CommandRateLimiter.from_yaml().max_rate.values()))
