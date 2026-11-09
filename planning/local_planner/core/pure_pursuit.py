"""Pure pursuit: pose + path -> ``LocoCommand``-style dict, through the B1 rate limiter.

Pure Python (stdlib + numpy). Registered in the local-planner candidate registry as ``pure_pursuit``. It is the
controller the H4 navigation eval runs on the frozen policy: the policy tracks (vx, vy, wz), so the controller only
has to produce a smooth, feasible twist.

Behaviour:

* goal tolerance (default 0.5 m): inside it the command is zero and ``at_goal`` is True;
* adaptive lookahead: ``clamp(lookahead_gain * speed, min_lookahead, max_lookahead)``;
* slowdown near the goal (``slow_radius_m``) and in sharp turns (the speed factor falls with |alpha|);
* the emitted command always goes through :class:`CommandRateLimiter`, so the controller cannot ask the policy for a
  step change the limiter would refuse.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from common.cerberus_eval.kinematics import wrap_angle
from common.cerberus_policies.command_limiter import CommandRateLimiter

ZERO_COMMAND = {
    "vx": 0.0,
    "vy": 0.0,
    "wz": 0.0,
    "body_pitch": 0.0,
    "body_height": 0.0,
    "has_body_pitch": False,
    "has_body_height": False,
}


@dataclass(frozen=True)
class PurePursuitConfig:
    lookahead_m: float = 0.6
    min_lookahead_m: float = 0.35
    max_lookahead_m: float = 1.2
    lookahead_gain_s: float = 0.8
    goal_tolerance_m: float = 0.5
    slow_radius_m: float = 1.2
    max_vx: float = 0.8
    max_wz: float = 0.9
    turn_slowdown_rad: float = 1.2
    min_speed_factor: float = 0.25
    dt: float = 0.02

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.goal_tolerance_m <= 0:
            raise ValueError("goal_tolerance_m must be positive")
        if not (0 < self.min_lookahead_m <= self.max_lookahead_m):
            raise ValueError("need 0 < min_lookahead <= max_lookahead")


def _closest_on_polyline(path: list[tuple[float, float]], xy: tuple[float, float]) -> tuple[float, float, float]:
    """``(arclength_at_closest, closest_x, closest_y)`` of the projection of ``xy`` onto ``path``."""
    best = (math.inf, 0.0, path[0][0], path[0][1])
    s = 0.0
    for (ax, ay), (bx, by) in itertools.pairwise(path):
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((xy[0] - ax) * dx + (xy[1] - ay) * dy) / seg2))
        px, py = ax + t * dx, ay + t * dy
        d = math.hypot(xy[0] - px, xy[1] - py)
        if d < best[0]:
            best = (d, s + t * math.sqrt(seg2), px, py)
        s += math.sqrt(seg2)
    return best[1], best[2], best[3]


def _point_at_arclength(path: list[tuple[float, float]], s: float) -> tuple[float, float]:
    """The point ``s`` metres along ``path`` (clamped to its ends)."""
    acc = 0.0
    for (ax, ay), (bx, by) in itertools.pairwise(path):
        seg = math.hypot(bx - ax, by - ay)
        if acc + seg >= s and seg > 0:
            t = (s - acc) / seg
            return (ax + t * (bx - ax), ay + t * (by - ay))
        acc += seg
    return path[-1]


class PurePursuit:
    """Stateful follower: one instance per robot, ``reset()`` between episodes."""

    def __init__(self, config: PurePursuitConfig | None = None, limiter: CommandRateLimiter | None = None) -> None:
        self.config = config or PurePursuitConfig()
        self.limiter = limiter
        self._last_speed = 0.0

    def reset(self) -> None:
        self._last_speed = 0.0
        if self.limiter is not None:
            self.limiter.reset()

    def at_goal(self, pose: tuple[float, float, float], goal: tuple[float, float]) -> bool:
        return math.hypot(pose[0] - goal[0], pose[1] - goal[1]) <= self.config.goal_tolerance_m

    def lookahead(self) -> float:
        return max(self.config.min_lookahead_m,
                   min(self.config.max_lookahead_m, self.config.lookahead_gain_s * self._last_speed + self.config.min_lookahead_m))

    def command(self, pose: tuple[float, float, float], path: list[tuple[float, float]], *, dt: float | None = None) -> dict:
        """The next ``LocoCommand`` dict for ``pose`` on ``path``."""
        cfg = self.config
        dt = cfg.dt if dt is None else float(dt)
        if len(path) < 2:
            return self._emit(ZERO_COMMAND, dt)
        goal = path[-1]
        if self.at_goal(pose, goal):
            self._last_speed = 0.0
            return self._emit(ZERO_COMMAND, dt)

        s0, _, _ = _closest_on_polyline(path, (pose[0], pose[1]))
        target = _point_at_arclength(path, s0 + self.lookahead())
        dx, dy = target[0] - pose[0], target[1] - pose[1]
        alpha = wrap_angle(math.atan2(dy, dx) - pose[2])
        dist = max(math.hypot(dx, dy), 1e-6)

        speed_factor = min(1.0, math.hypot(goal[0] - pose[0], goal[1] - pose[1]) / cfg.slow_radius_m)
        speed_factor *= max(cfg.min_speed_factor, 1.0 - abs(alpha) / cfg.turn_slowdown_rad)
        v = cfg.max_vx * max(0.0, min(1.0, speed_factor))
        wz = 2.0 * v * math.sin(alpha) / dist
        wz = max(-cfg.max_wz, min(cfg.max_wz, wz))
        self._last_speed = v
        return self._emit({**ZERO_COMMAND, "vx": v, "wz": wz}, dt)

    def _emit(self, command: dict, dt: float) -> dict:
        if self.limiter is None:
            return dict(command)
        return self.limiter.apply(command, dt)
