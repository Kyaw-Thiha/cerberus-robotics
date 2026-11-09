"""Scripted locomotion command source: a constant LocoCommand taken from config. Pure Python.
Produces B1 (LocoCommand). Select with ``get_backend("loco", "scripted_loco", config={"vx": 0.5})``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from common.cerberus_policies.base import Boundary, PolicyBackend, Role, make_loco_command


class ScriptedLocoBackend(PolicyBackend):
    boundary = Boundary.B1_LOCO
    role = Role.PRODUCER
    required_sensors = ()
    rate_hz = 50.0

    def __init__(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        wz: float = 0.0,
        body_pitch: float | None = None,
        body_height: float | None = None,
    ) -> None:
        self._command = make_loco_command(vx, vy, wz, body_pitch, body_height)
        self._loaded = False

    def load(self, checkpoint_path: str | None, embodiment_cfg: Any) -> None:
        if checkpoint_path is not None:
            raise ValueError("scripted_loco has no checkpoint; pass checkpoint_path=None")
        self._loaded = True

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("scripted_loco: call load() before act()")
        return dict(self._command)
