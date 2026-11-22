"""Null locomotion command source: always commands zero velocity. Pure Python; used in tests and as
the safe stand-still source. Produces B1 (LocoCommand)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from common.cerberus_policies.base import (
    Boundary,
    PolicyBackend,
    Role,
    make_loco_command,
)


class NullLocoBackend(PolicyBackend):
    boundary = Boundary.B1_LOCO
    role = Role.PRODUCER
    required_sensors = ()
    rate_hz = 50.0

    def __init__(self) -> None:
        self._loaded = False

    def load(self, checkpoint_path: str | None, embodiment_cfg: Any) -> None:
        if checkpoint_path is not None:
            raise ValueError("null_loco has no checkpoint; pass checkpoint_path=None")
        self._loaded = True

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("null_loco: call load() before act()")
        return make_loco_command()
