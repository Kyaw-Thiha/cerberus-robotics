"""JEPLO as a B1 locomotion backend: the authors' released ONNX pair driven by OUR sensor simulation.

This package is GPL-3.0 (see LICENSE) and optional: the core never imports it. Register it explicitly::

    from common.cerberus_policies.registry import register_backend
    from plugins.jeplo.backend import JeploBackend
    register_backend("loco", "jeplo", JeploBackend)

The released model is split into a 10 Hz sensor estimator (JEPA encoder + recurrent hidden) and a 50 Hz policy::

    sensor_estimator.onnx: prop_hist[1,450] + depth_stack[1,2,25,60] + hidden_in[1,512]
                           -> feature[1,512], hidden_out[1,512], depth_latent[1,64], prop_hist_latent[1,32]
    policy.onnx:           proprio_hist[1,450] + high_feat[1,512] -> actions[1,12]

Observation (45, fixed scales, newest first in a 10-frame history):
    base_ang_vel_b * 0.25 (3) | projected_gravity_b (3) | command vx,vy * 2.0 (2) | command wz * 0.25 (1) |
    joint_pos - default (12) | joint_vel * 0.05 (12) | previous actions * 0.1 (12)

No height scan, no ground-truth terrain: ``act`` refuses an observation carrying ``height_scan``.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from common.cerberus_policies.base import Boundary, PolicyBackend

JOINT_ORDER = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)
DEFAULT_JOINT_POS = {
    "FL_hip_joint": 0.1, "FR_hip_joint": -0.1, "RL_hip_joint": 0.1, "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
}
ACTION_SCALE = 0.25
HISTORY = 10
ESTIMATOR_EVERY = 5          # 50 Hz policy / 5 = 10 Hz sensor estimator
DEPTH_STACK = 2
OBS_CLIP = 100.0


class JeploObservationError(ValueError):
    pass


class JeploBackend(PolicyBackend):
    """The authors' released JEPLO checkpoint behind the B1 boundary, with our sensor simulation."""

    boundary = Boundary.B1_LOCO
    required_sensors: ClassVar[tuple[str, ...]] = ("lidar", "imu", "joint_encoders")
    rate_hz = 50.0

    def __init__(self) -> None:
        self._estimator = None
        self._policy = None
        self._hidden = np.zeros((1, 512), dtype=np.float32)
        self._feature = np.zeros((1, 512), dtype=np.float32)
        self._history: deque[np.ndarray] = deque(maxlen=HISTORY)
        self._depth_stack: deque[np.ndarray] = deque(maxlen=DEPTH_STACK)
        self._last_actions = np.zeros(12, dtype=np.float32)
        self._step = 0
        self._command = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        self._limiter = None
        self.estimator_calls = 0

    # -- loading ---------------------------------------------------------------------------------
    def load(self, checkpoint_path: str | None, embodiment_cfg: Any = None) -> None:
        if checkpoint_path is None:
            raise ValueError("jeplo needs the exported directory or sensor_estimator.onnx path")
        path = Path(checkpoint_path)
        if path.is_dir():
            estimator_path, policy_path = path / "sensor_estimator.onnx", path / "policy.onnx"
        else:
            estimator_path, policy_path = path, path.with_name("policy.onnx")
        for required in (estimator_path, policy_path):
            if not required.is_file():
                raise FileNotFoundError(f"JEPLO ONNX file missing: {required}")
        import onnxruntime as ort

        self._estimator = ort.InferenceSession(str(estimator_path), providers=["CPUExecutionProvider"])
        self._policy = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        est_inputs = {item.name: item.shape for item in self._estimator.get_inputs()}
        if est_inputs.get("prop_hist") != [1, 450] or est_inputs.get("depth_stack") != [1, 2, 25, 60]:
            raise ValueError(f"unexpected sensor_estimator inputs: {est_inputs}")
        pol_inputs = {item.name: item.shape for item in self._policy.get_inputs()}
        if pol_inputs.get("proprio_hist") != [1, 450] or pol_inputs.get("high_feat") != [1, 512]:
            raise ValueError(f"unexpected policy inputs: {pol_inputs}")
        self.reset()

    def set_command(self, command, *, dt: float = 0.02) -> dict[str, float]:
        """Rate-limit and range-clip a B1 command before it enters the observation (contract limiter)."""
        if self._limiter is None:
            from common.cerberus_eval.scripted_path import contract_limiter

            self._limiter = contract_limiter()
        self._command = self._limiter.apply({"vx": float(command.get("vx", 0.0)),
                                             "vy": float(command.get("vy", 0.0)),
                                             "wz": float(command.get("wz", 0.0))}, dt)
        return dict(self._command)

    def reset(self) -> None:
        self._limiter = None
        self._command = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        self._hidden[:] = 0.0
        self._feature[:] = 0.0
        self._history.clear()
        self._depth_stack.clear()
        self._last_actions[:] = 0.0
        self._step = 0

    # -- inference -------------------------------------------------------------------------------
    def act(self, observation) -> dict[str, Any]:
        if self._estimator is None or self._policy is None:
            raise RuntimeError("jeplo: call load() before act()")
        if "height_scan" in observation:
            raise JeploObservationError("a height scan reached the JEPLO observation; that is privileged data")
        depth = observation.get("depth")
        if depth is None:
            depth = np.ones((25, 60), dtype=np.float32)
        depth = np.asarray(depth, dtype=np.float32)
        if depth.shape != (25, 60):
            raise JeploObservationError(f"depth must be (25, 60), got {depth.shape}")

        proprio = self._proprio(observation)
        self._history.appendleft(proprio)
        hist = np.concatenate(list(self._history)) if self._history else proprio
        if hist.size < 45 * HISTORY:
            pad = np.zeros(45 * HISTORY - hist.size, dtype=np.float32)
            hist = np.concatenate([hist, pad])

        if self._step % ESTIMATOR_EVERY == 0:
            self._depth_stack.append(depth)
            while len(self._depth_stack) < DEPTH_STACK:
                self._depth_stack.appendleft(depth.copy())
            stack = np.stack(list(self._depth_stack), axis=0)[None]  # (1, 2, 25, 60)
            feature, hidden, *_ = self._estimator.run(None, {
                "prop_hist": hist[None].astype(np.float32),
                "depth_stack": stack.astype(np.float32),
                "hidden_in": self._hidden.astype(np.float32),
            })
            self._feature = np.asarray(feature, dtype=np.float32).reshape(1, 512)
            self._hidden = np.asarray(hidden, dtype=np.float32).reshape(1, 512)
            self.estimator_calls += 1

        raw = self._policy.run(None, {
            "proprio_hist": hist[None].astype(np.float32),
            "high_feat": self._feature.astype(np.float32),
        })[0]
        actions = np.asarray(raw, dtype=np.float32).reshape(12)
        if not np.isfinite(actions).all():
            raise FloatingPointError("JEPLO returned a non-finite action")
        actions = np.clip(actions, -OBS_CLIP, OBS_CLIP)
        self._last_actions = actions
        self._step += 1
        return {"actions": actions, "joint_targets": self.actions_to_joint_targets(actions)}

    def _proprio(self, observation) -> np.ndarray:
        command = observation.get("command") or self._command
        vx, vy, wz = float(command.get("vx", 0.0)), float(command.get("vy", 0.0)), float(command.get("wz", 0.0))
        ang = np.asarray(observation["base_ang_vel"], dtype=np.float32).reshape(3) * 0.25
        grav = np.asarray(observation["projected_gravity"], dtype=np.float32).reshape(3)
        cmd = np.array([vx * 2.0, vy * 2.0, wz * 0.25], dtype=np.float32)
        joint_pos = self._by_name(observation["joint_pos"])
        joint_vel = self._by_name(observation["joint_vel"])
        pos = np.array([joint_pos[name] - DEFAULT_JOINT_POS[name] for name in JOINT_ORDER], dtype=np.float32)
        vel = np.array([joint_vel[name] for name in JOINT_ORDER], dtype=np.float32) * 0.05
        vector = np.concatenate([ang, grav, cmd, pos, vel, self._last_actions * 0.1]).astype(np.float32)
        if not np.isfinite(vector).all():
            raise FloatingPointError("JEPLO proprioception has a non-finite value")
        return np.clip(vector, -OBS_CLIP, OBS_CLIP)

    @staticmethod
    def _by_name(values) -> dict[str, float]:
        if isinstance(values, dict):
            missing = [name for name in JOINT_ORDER if name not in values]
            if missing:
                raise JeploObservationError(f"joint observation is missing {missing}")
            return {name: float(values[name]) for name in JOINT_ORDER}
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.size != 12:
            raise JeploObservationError(f"joint array must be 12 long in the JEPLO order, got {array.size}")
        return dict(zip(JOINT_ORDER, (float(v) for v in array)))

    def actions_to_joint_targets(self, actions) -> dict[str, float]:
        a = np.asarray(actions, dtype=np.float32).reshape(-1)
        if a.size != 12:
            raise ValueError(f"expected 12 actions, got {a.size}")
        if not math.isfinite(float(np.abs(a).sum())):
            raise FloatingPointError("non-finite action")
        return {name: DEFAULT_JOINT_POS[name] + ACTION_SCALE * float(a[i]) for i, name in enumerate(JOINT_ORDER)}
