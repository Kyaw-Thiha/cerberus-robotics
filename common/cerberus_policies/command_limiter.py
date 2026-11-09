"""B1 command slew-rate limiter: bound how fast a velocity command may change.

Motivation (H2, `docs/reports/phase05_adversarial_report.md`): the Phase 0.5 adversary breaks the
locomotion candidate through the B1 velocity command — it pins an axis at its clamp and flips sign
~3.4 times/s, a bandwidth the policy never saw (training resamples commands every 10 s). Bounding the
command's rate of change is the hypothesised mitigation. **Its effect is untested**: nothing here
claims the gate would pass behind it, and the defaults shipped in
`common/cerberus_policies/config/command_limiter.yaml` are placeholders, not tuned values.

This is deployment-side and unrelated to `policy/locomotion/adversarial_robustness/attack_math.py`'s
``RateLimitedChannel``: that one *bounds the adversary* (2-D force/command channel, a fraction of the
channel peak per sim step, inside a frozen gate whose artifacts must stay reproducible). This one
bounds a *real* B1 command: three axes, per-axis rates in physical units per second, an explicit
``dt``, optional body fields, and no force. The two are kept separate on purpose — importing
attack-side code here would couple deployment to the adversarial package and its sim-step semantics.

Pure Python (stdlib only): importable without torch, ROS or Isaac.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: axes with a slew limit; units are m/s^2 for vx/vy and rad/s^2 for wz
RATE_AXES = ("vx", "vy", "wz")
RATE_UNITS = {"vx": "m/s^2", "vy": "m/s^2", "wz": "rad/s^2"}
#: axes that may also carry an absolute clamp; body fields are clamped but never rate-limited
CLAMP_AXES = ("vx", "vy", "wz", "body_pitch", "body_height")
OPTIONAL_AXES = ("body_pitch", "body_height")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "command_limiter.yaml"


class CommandLimiterConfigError(ValueError):
    pass


@dataclass
class CommandRateLimiter:
    """Per-axis slew-rate limiter and clamp for a plain ``LocoCommand`` dict.

    ``max_rate``: maximum change per second, per rate axis (m/s^2, m/s^2, rad/s^2).
    ``max_abs``: absolute clamp per axis (m/s, m/s, rad/s; radians for the body fields).

    ``apply`` is stateful: it moves each axis from the previously emitted value toward the requested
    one by at most ``max_rate[axis] * dt``. The first ``apply`` after ``reset()`` adopts the clamped
    request outright -- there is no previous command to slew from, and inventing a start-up transient
    (e.g. a mandatory ramp from zero) is a different behaviour than "bound how fast the command
    changes". ``body_pitch`` / ``body_height`` pass through clamped only, never rate-limited, and
    their ``has_*`` flags are copied untouched.
    """

    max_rate: Mapping[str, float]
    max_abs: Mapping[str, float]
    _state: dict[str, float] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.max_rate = dict(self.max_rate)
        self.max_abs = dict(self.max_abs)
        missing = [a for a in RATE_AXES if a not in self.max_rate]
        if missing:
            raise CommandLimiterConfigError(f"max_rate is missing axes {missing}")
        known_rate = set(RATE_AXES) | set(OPTIONAL_AXES)
        extra = sorted(set(self.max_rate) - known_rate)
        if extra:
            raise CommandLimiterConfigError(f"max_rate has unknown axes {extra}; known: {sorted(known_rate)}")
        extra = sorted(set(self.max_abs) - set(CLAMP_AXES))
        if extra:
            raise CommandLimiterConfigError(f"max_abs has unknown axes {extra}; known: {sorted(CLAMP_AXES)}")
        missing = [a for a in RATE_AXES if a not in self.max_abs]
        if missing:
            raise CommandLimiterConfigError(f"max_abs is missing axes {missing}")
        for axis, value in self.max_rate.items():
            if not _is_positive_finite(value):
                raise CommandLimiterConfigError(f"max_rate[{axis!r}] must be a positive finite number, got {value!r}")
        for axis, value in self.max_abs.items():
            if not _is_positive_finite(value):
                raise CommandLimiterConfigError(f"max_abs[{axis!r}] must be a positive finite number, got {value!r}")

    # -- construction -------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> CommandRateLimiter:
        """Build from a mapping shaped like ``config/command_limiter.yaml``."""
        if not isinstance(config, Mapping) or "max_rate" not in config or "max_abs" not in config:
            raise CommandLimiterConfigError("config needs 'max_rate' and 'max_abs' mappings")
        return cls(max_rate=config["max_rate"], max_abs=config["max_abs"])

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> CommandRateLimiter:
        """Build from a YAML file; the checked-in file holds untested placeholder values."""
        return cls.from_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    # -- runtime ------------------------------------------------------------------------------
    def reset(self) -> None:
        """Forget the previous command; the next ``apply`` adopts its (clamped) target."""
        self._state = None

    def apply(self, command: Mapping[str, Any], dt: float) -> dict[str, Any]:
        """Return a copy of ``command`` with every axis slew-limited toward its clamp.

        ``dt`` is the time since the previous command (seconds). Raises ``ValueError`` for a
        non-finite/non-positive ``dt`` or a non-finite command value.
        """
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError(f"dt must be a positive finite number of seconds, got {dt!r}")

        out = dict(command)
        for axis in RATE_AXES:
            if axis not in out:
                raise ValueError(f"command is missing {axis!r}")
        for axis in CLAMP_AXES:
            if axis in out:
                out[axis] = self._clamp(axis, _finite(out[axis], axis))
        for axis in OPTIONAL_AXES:
            if axis in out:
                out[axis] = self._clamp(axis, _finite(out[axis], axis))

        previous = self._state
        state: dict[str, float] = {}
        for axis in RATE_AXES:
            target = out[axis]
            if previous is None:
                state[axis] = target
                continue
            delta = target - previous[axis]
            step = self.max_rate[axis] * dt
            state[axis] = previous[axis] + max(-step, min(step, delta))
        self._state = state
        for axis in RATE_AXES:
            out[axis] = state[axis]
        return out

    def _clamp(self, axis: str, value: float) -> float:
        limit = self.max_abs.get(axis)
        return value if limit is None else max(-limit, min(limit, value))


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"command[{name!r}] must be a finite number, got {value!r}")
    return float(value)


def _is_positive_finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0
