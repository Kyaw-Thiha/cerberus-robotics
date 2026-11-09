"""Pure-Python core of the locomotion inference node: observation assembly, B1 rate limiting,
policy call and action -> joint-target mapping. No rclpy here, so it is unit-testable locally.

The rclpy wrapper (``inference_node.py``) only wires topics/parameters and hands this core plain
values. Term order and the 48/235 observation layout come from
``observation_assembly`` (the training run's params); the ONNX backend, its contract and the
limiter live in ``common/cerberus_policies`` and are resolved by the wrapper, not here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from locomotion_policy.observation_assembly import (
    SHAPE_BLIND,
    SHAPE_PERCEPTIVE,
    ProprioSample,
    apply_action_postprocessing,
    assemble_observation,
)

LOCO_COMMAND_FIELDS = ("vx", "vy", "wz", "body_pitch", "body_height", "has_body_pitch", "has_body_height")


def zero_loco_command() -> dict[str, Any]:
    return {
        "vx": 0.0,
        "vy": 0.0,
        "wz": 0.0,
        "body_pitch": 0.0,
        "body_height": 0.0,
        "has_body_pitch": False,
        "has_body_height": False,
    }


@dataclass(frozen=True)
class PolicySpec:
    """Static policy configuration resolved from launch parameters / the observation contract."""

    joint_names: tuple[str, ...]
    default_joint_pos: tuple[float, ...]
    action_scale: float = 0.25
    observation_shape: str = SHAPE_PERCEPTIVE
    height_scan_size: int = 187

    def __post_init__(self) -> None:
        if len(self.joint_names) != len(self.default_joint_pos):
            raise ValueError(
                f"{len(self.joint_names)} joint names vs {len(self.default_joint_pos)} default positions"
            )
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names contains duplicates; ordering must be unambiguous")
        if not self.action_scale > 0:
            raise ValueError(f"action_scale must be positive, got {self.action_scale}")
        if self.observation_shape not in (SHAPE_BLIND, SHAPE_PERCEPTIVE):
            raise ValueError(f"observation_shape must be {SHAPE_BLIND!r} or {SHAPE_PERCEPTIVE!r}, got {self.observation_shape!r}")

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


@dataclass(frozen=True)
class StepOutput:
    observation: np.ndarray
    actions: np.ndarray
    joint_targets: dict[str, float] = field(default_factory=dict)


PolicyFn = Callable[[np.ndarray], Any]


class LocomotionCore:
    """One control step from plain values to joint targets keyed by name.

    ``policy_fn`` receives the assembled observation vector (float32, shape (48,) or (235,)) and
    returns 12 raw actions. ``limiter`` is optional and only needs ``apply(command, dt)`` /
    ``reset()`` (``common.cerberus_policies.command_limiter.CommandRateLimiter``).
    """

    def __init__(self, spec: PolicySpec, policy_fn: PolicyFn, limiter: Any = None) -> None:
        self.spec = spec
        self.policy_fn = policy_fn
        self.limiter = limiter
        self._last_action = np.zeros(spec.num_joints, dtype=np.float32)

    def reset(self) -> None:
        self._last_action = np.zeros(self.spec.num_joints, dtype=np.float32)
        if self.limiter is not None:
            self.limiter.reset()

    # -- B1 command --------------------------------------------------------------------------
    def command(self, command: Mapping[str, Any], dt: float) -> dict[str, Any]:
        """Validate a LocoCommand dict and run it through the rate limiter (if configured)."""
        missing = [k for k in LOCO_COMMAND_FIELDS if k not in command]
        if missing:
            raise ValueError(f"LocoCommand dict is missing {missing}")
        out = {k: command[k] for k in LOCO_COMMAND_FIELDS}
        for axis in ("vx", "vy", "wz"):
            if isinstance(out[axis], bool) or not isinstance(out[axis], (int, float)):
                raise ValueError(f"command[{axis!r}] must be a number, got {out[axis]!r}")
        if self.limiter is not None:
            out = self.limiter.apply(out, dt)
        return out

    # -- policy step -------------------------------------------------------------------------
    def step(
        self,
        *,
        base_lin_vel: Any,
        base_ang_vel: Any,
        projected_gravity: Any,
        joint_positions: Mapping[str, float],
        joint_velocities: Mapping[str, float],
        height_scan: Any,
        command: Mapping[str, Any],
    ) -> StepOutput:
        joint_pos = self._ordered(joint_positions, "joint position")
        joint_vel = self._ordered(joint_velocities, "joint velocity")
        gravity = np.asarray(projected_gravity, dtype=np.float32).reshape(3)
        grav_norm = float(np.linalg.norm(gravity))
        if grav_norm > 0:
            gravity = gravity / grav_norm

        proprio = ProprioSample(
            base_lin_vel=np.asarray(base_lin_vel, dtype=np.float32).reshape(3),
            base_ang_vel=np.asarray(base_ang_vel, dtype=np.float32).reshape(3),
            projected_gravity=gravity.astype(np.float32),
            velocity_commands=np.array([command["vx"], command["vy"], command["wz"]], dtype=np.float32),
            joint_pos_rel=(joint_pos - np.asarray(self.spec.default_joint_pos, dtype=np.float32)).astype(np.float32),
            joint_vel_rel=joint_vel,
            last_action=self._last_action,
        )
        scan = None
        if self.spec.observation_shape == SHAPE_PERCEPTIVE:
            if height_scan is None:
                raise ValueError("perceptive observation shape requires a height scan")
            scan = np.asarray(height_scan, dtype=np.float32).reshape(-1)
            if scan.size != self.spec.height_scan_size:
                raise ValueError(f"height scan has {scan.size} cells, expected {self.spec.height_scan_size}")
        obs = assemble_observation(self.spec.observation_shape, proprio, scan)

        actions = np.asarray(self.policy_fn(obs), dtype=np.float32).reshape(-1)
        if actions.size != self.spec.num_joints:
            raise ValueError(f"policy returned {actions.size} actions, expected {self.spec.num_joints}")
        self._last_action = actions
        targets = apply_action_postprocessing(
            actions, np.asarray(self.spec.default_joint_pos, dtype=np.float32), scale=self.spec.action_scale
        )
        return StepOutput(observation=obs, actions=actions, joint_targets=dict(zip(self.spec.joint_names, map(float, targets))))

    def _ordered(self, values: Mapping[str, float], what: str) -> np.ndarray:
        missing = [n for n in self.spec.joint_names if n not in values]
        extra = sorted(set(values) - set(self.spec.joint_names))
        if missing or extra:
            raise ValueError(f"{what} names do not match the policy's joint list: missing={missing} unexpected={extra}")
        return np.array([float(values[n]) for n in self.spec.joint_names], dtype=np.float32)


def build_limiter(config_path: str | Path, repo_root: str | Path | None = None) -> Any:
    """Load ``CommandRateLimiter`` from a YAML config (the shipped file holds placeholders).

    Imported lazily: this package must stay importable from a bare repo checkout, and the limiter
    lives in ``common/cerberus_policies``, which the wrapper puts on ``sys.path`` first.
    """
    from common.cerberus_policies.command_limiter import CommandRateLimiter

    return CommandRateLimiter.from_yaml(Path(config_path))
