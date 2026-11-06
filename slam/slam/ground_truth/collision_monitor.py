"""Collision and clearance bookkeeping for one SLAM eval mission (pure numpy, no Isaac imports).

Why this exists: the scripted mission is open loop, so a run can push into a wall or furniture and the
resulting drift numbers would measure the collision, not the SLAM backend. Every run therefore records
whether the robot touched the scene, and ``valid_for_scoring`` says whether its ATE/RPE may be reported.

Two independent signals, both advisory until calibrated on a real run:

* **Contact** (the verdict): the stock Go2 task already carries a ``contact_forces`` ContactSensor on every
  robot body. The sensor is unfiltered, so the floor contributes too: feet are expected to touch it and are
  ignored. Any other body (base, hip, thigh, calf) above ``CONTACT_FORCE_THRESHOLD_N`` counts as a collision.
  The threshold is provisional, and a policy that crouches low may rub a calf on the floor; the report lists
  which bodies were involved so a false positive is recognisable.
* **Clearance** (informational only): the smallest horizontal distance from the robot base to scene geometry
  seen by the LiDAR inside the robot's height band. The floor is excluded by the band, which is why the raw
  minimum range is not used. It is measured from the base centre (the Go2 half-length is about 0.35 m) and
  only when the rig has a LiDAR, so it is ``None`` for the camera-only rigs. It never changes the verdict.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

CONTACT_FORCE_THRESHOLD_N = 5.0
"""Net contact force on a non-foot body above which the body counts as touching the scene (provisional)."""

IGNORED_BODY_SUFFIXES = ("foot",)
"""Bodies expected to touch the floor, matched case-insensitively (``FL_foot`` on the Go2 USD)."""

CLEARANCE_BAND_ABOVE_FLOOR_M = (0.05, 0.45)
"""LiDAR hits between these heights above the floor are treated as obstacles (below: floor and rugs)."""


def offending_bodies(
    body_names: Sequence[str],
    force_norms_n: Sequence[float],
    threshold_n: float = CONTACT_FORCE_THRESHOLD_N,
    ignore_suffixes: Sequence[str] = IGNORED_BODY_SUFFIXES,
) -> list[str]:
    """Names of the non-ignored bodies whose contact force norm exceeds ``threshold_n`` (NaN counts as offending)."""
    if len(body_names) != len(force_norms_n):
        raise ValueError(f"{len(body_names)} body names but {len(force_norms_n)} force values")
    ignored = tuple(suffix.lower() for suffix in ignore_suffixes)
    return [
        name
        for name, force in zip(body_names, force_norms_n, strict=True)
        if not name.lower().endswith(ignored) and not float(force) <= threshold_n
    ]


def band_clearance_m(
    hits_w: np.ndarray,
    base_xy: Sequence[float],
    floor_z: float,
    band_above_floor_m: tuple[float, float] = CLEARANCE_BAND_ABOVE_FLOOR_M,
) -> float | None:
    """Smallest horizontal distance from ``base_xy`` to a finite world-frame ray hit inside the height band.

    ``hits_w`` is (M, 3) with ``inf``/NaN for misses. Returns ``None`` when nothing lies in the band.
    """
    hits = np.asarray(hits_w, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(hits).all(axis=-1)
    height = hits[:, 2] - floor_z
    in_band = finite & (height >= band_above_floor_m[0]) & (height <= band_above_floor_m[1])
    if not in_band.any():
        return None
    delta = hits[in_band, :2] - np.asarray(base_xy, dtype=np.float64)
    return float(np.hypot(delta[:, 0], delta[:, 1]).min())


@dataclass
class CollisionMonitor:
    """Accumulates per-step contact and clearance observations into one report block."""

    threshold_n: float = CONTACT_FORCE_THRESHOLD_N
    contact_seen: bool = False
    collision_steps: int = 0
    collision_events: int = 0
    first_collision_t_s: float | None = None
    bodies: set[str] = field(default_factory=set)
    min_clearance_m: float | None = None
    clearance_samples: int = 0
    _in_collision: bool = False

    def update(self, t_s: float, offenders: Sequence[str] | None, clearance_m: float | None = None) -> None:
        """Record one step. ``offenders`` is ``None`` when the contact sensor does not exist (verdict unknown)."""
        if clearance_m is not None and math.isfinite(clearance_m):
            self.clearance_samples += 1
            self.min_clearance_m = clearance_m if self.min_clearance_m is None else min(self.min_clearance_m, clearance_m)
        if offenders is None:
            return
        self.contact_seen = True
        touching = len(offenders) > 0
        if touching:
            self.collision_steps += 1
            self.bodies.update(offenders)
            if not self._in_collision:
                self.collision_events += 1
                if self.first_collision_t_s is None:
                    self.first_collision_t_s = t_s
        self._in_collision = touching

    def invalid_reasons(self) -> list[str]:
        reasons = []
        if not self.contact_seen:
            reasons.append("no_contact_sensor")
        if self.collision_events > 0:
            reasons.append("collision")
        return reasons

    def summary(self) -> dict:
        reasons = self.invalid_reasons()
        return {
            "contact_sensor": self.contact_seen,
            "contact_threshold_n": self.threshold_n,
            "collision_steps": self.collision_steps,
            "collision_events": self.collision_events,
            "first_collision_t_s": self.first_collision_t_s,
            "bodies_in_collision": sorted(self.bodies),
            "min_clearance_m": self.min_clearance_m,
            "clearance_samples": self.clearance_samples,
            "valid_for_scoring": not reasons,
            "invalid_reasons": reasons,
        }
