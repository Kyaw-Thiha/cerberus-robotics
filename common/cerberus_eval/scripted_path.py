"""Scripted velocity paths: waypoints -> a time series of ``LocoCommand`` dicts (pure Python, no torch/Isaac).

Used as the pod smoke test (R2): it drives the frozen policy through the env's command hook with a known open-loop
schedule, validating the env, the hook and the video helper before any planner is involved. It is NOT a controller:
it does not look at the robot's pose.

Every emitted command passes through :class:`common.cerberus_policies.command_limiter.CommandRateLimiter`, so the
schedule's ramps are the limiter's ramps and the per-axis clamps are the contract's command ranges
(``contracts/heightscan_ppo_v1.yaml`` ``commands.ranges``), not numbers invented here.

Determinism: :func:`random_waypoints` takes a seed and uses a private ``random.Random``; the rest is arithmetic.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import yaml

from common.cerberus_eval.kinematics import Pose, integrate, wrap_angle
from common.cerberus_policies.command_limiter import DEFAULT_CONFIG_PATH, CommandRateLimiter

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "cerberus_policies" / "contracts" / "heightscan_ppo_v1.yaml"
ZERO_COMMAND = {
    "vx": 0.0,
    "vy": 0.0,
    "wz": 0.0,
    "body_pitch": 0.0,
    "body_height": 0.0,
    "has_body_pitch": False,
    "has_body_height": False,
}


def contract_command_limits(contract_path: str | Path = CONTRACT_PATH) -> dict[str, float]:
    """Per-axis absolute limits ``{vx, vy, wz}`` from the observation contract's ``commands.ranges``."""
    ranges = yaml.safe_load(Path(contract_path).read_text(encoding="utf-8"))["commands"]["ranges"]
    return {
        "vx": max(abs(v) for v in ranges["vx_mps"]),
        "vy": max(abs(v) for v in ranges["vy_mps"]),
        "wz": max(abs(v) for v in ranges["wz_radps"]),
    }


def contract_limiter(contract_path: str | Path = CONTRACT_PATH) -> CommandRateLimiter:
    """The shipped limiter rates with the contract's clamps (they agree today; the contract is the source of truth)."""
    config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    config["max_abs"] = {**config["max_abs"], **contract_command_limits(contract_path)}
    return CommandRateLimiter.from_config(config)


@dataclass(frozen=True)
class ScriptedPath:
    """A generated schedule: ``commands[i]`` is applied for ``dt`` seconds."""

    commands: list[dict]
    dt: float
    start_pose: Pose
    waypoints: list[tuple[float, float]]
    expected_end_pose: Pose  # integrated from the LIMITED commands (what a perfect tracker would reach)

    @property
    def duration_s(self) -> float:
        return len(self.commands) * self.dt


def random_waypoints(
    seed: int,
    n: int = 3,
    leg_m: tuple[float, float] = (1.0, 2.0),
    start: tuple[float, float] = (0.0, 0.0),
    max_turn_rad: float = math.pi / 2,
    start_yaw: float = 0.0,
) -> list[tuple[float, float]]:
    """``n`` waypoints of a random polyline from ``start``; each leg turns at most ``max_turn_rad``. Deterministic."""
    rng = random.Random(seed)
    x, y = start
    heading = start_yaw
    points = []
    for _ in range(n):
        heading = wrap_angle(heading + rng.uniform(-max_turn_rad, max_turn_rad))
        length = rng.uniform(*leg_m)
        x, y = x + length * math.cos(heading), y + length * math.sin(heading)
        points.append((round(x, 6), round(y, 6)))
    return points


def waypoints_to_commands(
    start_pose: Pose,
    waypoints: list[tuple[float, float]],
    *,
    dt: float = 0.02,
    speed: float = 0.5,
    turn_rate: float = 0.5,
    settle_s: float = 1.0,
    hold_s: float = 0.5,
    limiter: CommandRateLimiter | None = None,
) -> ScriptedPath:
    """Turn in place to face each waypoint, then walk straight to it; limiter ramps every transition.

    ``speed`` / ``turn_rate`` are clipped to the contract limits (never exceeded, whatever the caller passes).
    A ``settle_s`` zero-command lead-in lets the policy stand before moving; ``hold_s`` of zeros ends the schedule.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    limiter = limiter if limiter is not None else contract_limiter()
    limits = {axis: limiter.max_abs[axis] for axis in ("vx", "wz")}
    speed = min(abs(speed), limits["vx"])
    turn_rate = min(abs(turn_rate), limits["wz"])
    if speed <= 0 or turn_rate <= 0:
        raise ValueError("speed and turn_rate must be positive")

    targets: list[tuple[float, float, float]] = []  # (vx, wz, duration_s)
    for _ in range(round(settle_s / dt)):
        targets.append((0.0, 0.0, dt))
    x, y, yaw = start_pose
    for wx, wy in waypoints:
        bearing = math.atan2(wy - y, wx - x)
        turn = wrap_angle(bearing - yaw)
        if abs(turn) > 1e-6:
            steps = max(1, math.ceil(abs(turn) / turn_rate / dt))
            wz = math.copysign(abs(turn) / (steps * dt), turn)  # lands exactly on the bearing (open loop)
            targets.extend([(0.0, wz, dt)] * steps)
        dist = math.hypot(wx - x, wy - y)
        steps = max(1, math.ceil(dist / speed / dt))
        targets.extend([(dist / (steps * dt), 0.0, dt)] * steps)
        x, y, yaw = wx, wy, bearing
    targets.extend([(0.0, 0.0, dt)] * round(hold_s / dt))

    limiter.reset()
    commands = []
    for vx, wz, _ in targets:
        commands.append(limiter.apply({**ZERO_COMMAND, "vx": vx, "wz": wz}, dt))
    # A hold of zero TARGETS is not a hold: the limiter still has to ramp down from the last speed. Ramp to a true
    # stop first, then hold, so the schedule really ends with the robot standing.
    guard = int(10.0 / dt)
    while commands and (commands[-1]["vx"] != 0.0 or commands[-1]["wz"] != 0.0):
        commands.append(limiter.apply(dict(ZERO_COMMAND), dt))
        guard -= 1
        if guard <= 0:
            raise RuntimeError("the limiter did not reach a zero command within 10 s of zero targets")
    commands.extend([limiter.apply(dict(ZERO_COMMAND), dt)] * round(hold_s / dt))
    end = integrate(start_pose, commands, dt)
    return ScriptedPath(commands=commands, dt=dt, start_pose=start_pose, waypoints=list(waypoints), expected_end_pose=end)


def scripted_path_for_seed(seed: int, start_pose: Pose = (0.0, 0.0, 0.0), **kwargs) -> ScriptedPath:
    """Convenience for the smoke test: a seeded random polyline from ``start_pose`` as a command schedule."""
    wps = random_waypoints(seed, start=(start_pose[0], start_pose[1]), start_yaw=start_pose[2])
    return waypoints_to_commands(start_pose, wps, **kwargs)
