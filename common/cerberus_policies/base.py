"""Policy backend interface: one swappable backend behind each of the three command boundaries.

B1 locomotion command (`LocoCommand`), B2 arm command (`ArmCommand`), B3 body command
(`BodyCommand`); message definitions live in `common/cerberus_msgs/msg/`. Everything else goes
through ``registry.get_backend`` and never imports a backend module directly.

Deliberately no torch/numpy/ROS/Isaac import at module level: backends import heavy deps inside
``load`` / ``act``, so this package imports anywhere (CI, a laptop, a ROS-less container).

Observation convention (not enforced by the ABC, so fused backends are not forced through a
pre-processed map): a plain dict, typically
``{"t": float, "proprio": {...}, "sensors": {sensor_id: raw_stream}, "command": <dict at this boundary>}``.
Modular backends read upstream module output (e.g. an elevation map) from ``observation``; fused
backends read raw streams from ``observation["sensors"]`` using the ids in ``required_sensors``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, ClassVar


class Boundary(str, Enum):
    B1_LOCO = "B1_LOCO"
    B2_ARM = "B2_ARM"
    B3_BODY = "B3_BODY"


class Role(str, Enum):
    """Which side of the boundary a backend sits on.

    CONSUMER: reads the boundary command (``observation["command"]``) and drives what is below it
    (e.g. a locomotion policy: B1 command in, joint targets out). PRODUCER: emits the boundary
    command itself (a planner, a scripted source, a VLA head). ``act`` returns a dict either way.
    """

    PRODUCER = "producer"
    CONSUMER = "consumer"


# Command dict conventions, mirroring the .msg definitions (headers are added by the ROS adapter).
LOCO_COMMAND_KEYS = ("vx", "vy", "wz", "body_pitch", "body_height", "has_body_pitch", "has_body_height")
ARM_COMMAND_KEYS = ("mode", "ee_poses", "joint_positions", "num_joints", "gripper", "dt")
BODY_COMMAND_KEYS = ("loco", "arm", "dt")
ARM_MODES = {"EE_POSE": 0, "EE_TRAJECTORY": 1, "JOINT_POSITION": 2}


class CommandShapeError(ValueError):
    pass


def make_loco_command(
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    body_pitch: float | None = None,
    body_height: float | None = None,
) -> dict[str, Any]:
    """Build a LocoCommand dict. ``None`` for an optional field -> value 0.0 with its has_* flag False."""
    return {
        "vx": float(vx),
        "vy": float(vy),
        "wz": float(wz),
        "body_pitch": 0.0 if body_pitch is None else float(body_pitch),
        "body_height": 0.0 if body_height is None else float(body_height),
        "has_body_pitch": body_pitch is not None,
        "has_body_height": body_height is not None,
    }


def check_command(boundary: Boundary, command: Mapping[str, Any]) -> None:
    """Raise CommandShapeError unless ``command`` has exactly the fields (and basic types) of the boundary's message."""
    if boundary is Boundary.B1_LOCO:
        _check_keys(command, LOCO_COMMAND_KEYS, "LocoCommand")
        for k in ("vx", "vy", "wz", "body_pitch", "body_height"):
            _check_number(command[k], k)
        for k in ("has_body_pitch", "has_body_height"):
            if not isinstance(command[k], bool):
                raise CommandShapeError(f"LocoCommand.{k} must be bool, got {type(command[k]).__name__}")
    elif boundary is Boundary.B2_ARM:
        _check_keys(command, ARM_COMMAND_KEYS, "ArmCommand")
        if command["mode"] not in ARM_MODES.values():
            raise CommandShapeError(f"ArmCommand.mode must be one of {sorted(ARM_MODES.values())}")
        for k in ("ee_poses", "joint_positions", "gripper"):
            if not isinstance(command[k], Sequence) or isinstance(command[k], str):
                raise CommandShapeError(f"ArmCommand.{k} must be a sequence")
        if command["mode"] == ARM_MODES["JOINT_POSITION"]:
            n = int(command["num_joints"])
            if n <= 0 or len(command["joint_positions"]) % n != 0:
                raise CommandShapeError("ArmCommand.joint_positions length must be a multiple of num_joints")
        _check_number(command["dt"], "dt")
    elif boundary is Boundary.B3_BODY:
        _check_keys(command, BODY_COMMAND_KEYS, "BodyCommand")
        check_command(Boundary.B1_LOCO, command["loco"])
        check_command(Boundary.B2_ARM, command["arm"])
        _check_number(command["dt"], "dt")


def _check_keys(command: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    missing = [k for k in keys if k not in command]
    extra = [k for k in command if k not in keys]
    if missing or extra:
        raise CommandShapeError(f"{name} dict mismatch: missing={missing} unexpected={extra}")


def _check_number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommandShapeError(f"{name} must be a number, got {type(value).__name__}")


class PolicyBackend(ABC):
    """Contract every policy backend must fulfil.

    Concrete subclasses set the class attributes below; ``__init_subclass__`` checks them so a
    backend that forgets one fails at import/definition time, not mid-run.
    """

    boundary: ClassVar[Boundary]
    role: ClassVar[Role] = Role.CONSUMER
    #: sensor ids, same naming as plans/sensor_setup_handoff.md / sensor_profiles.yaml
    #: (lidar, front_camera, rear_camera, wrist_camera, imu) plus joint_encoders. Fused backends
    #: list the raw streams they consume here.
    required_sensors: ClassVar[Sequence[str]] = ()
    rate_hz: ClassVar[float]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # ABCMeta computes __abstractmethods__ only after __init_subclass__ runs, so detect
        # "still abstract" (an intermediate base class) directly.
        if any(getattr(getattr(cls, n, None), "__isabstractmethod__", False) for n in dir(cls)):
            return
        if not isinstance(getattr(cls, "boundary", None), Boundary):
            raise TypeError(f"{cls.__name__}.boundary must be a Boundary")
        if not isinstance(cls.role, Role):
            raise TypeError(f"{cls.__name__}.role must be a Role")
        rate = getattr(cls, "rate_hz", None)
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
            raise TypeError(f"{cls.__name__}.rate_hz must be a positive number")
        sensors = cls.required_sensors
        if isinstance(sensors, str) or not all(isinstance(s, str) for s in sensors):
            raise TypeError(f"{cls.__name__}.required_sensors must be a sequence of sensor id strings")

    @abstractmethod
    def load(self, checkpoint_path: str | None, embodiment_cfg: Any) -> None:
        """Load weights/config. ``embodiment_cfg`` is an EmbodimentCfg (joints addressed by name)."""

    @abstractmethod
    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """One control step: observation dict -> command dict (plain Python/NumPy types)."""
