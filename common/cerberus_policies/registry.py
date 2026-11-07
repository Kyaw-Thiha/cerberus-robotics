"""Policy backend registry. Backends are chosen explicitly per slot (argument or the slot's env var)
-- there is deliberately no silent default, so a run never quietly lands on the wrong policy.

Slots and env vars: loco -> LOCO_BACKEND (B1), arm -> ARM_BACKEND (B2), body -> BODY_BACKEND (B3).
Factories may be lazy "module:Class" strings so heavy backends are only imported when first
requested.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any

from common.cerberus_policies.base import Boundary, PolicyBackend

SLOT_ENV: dict[str, str] = {"loco": "LOCO_BACKEND", "arm": "ARM_BACKEND", "body": "BODY_BACKEND"}
SLOT_BOUNDARY: dict[str, Boundary] = {
    "loco": Boundary.B1_LOCO,
    "arm": Boundary.B2_ARM,
    "body": Boundary.B3_BODY,
}

Factory = str | Callable[..., PolicyBackend]

_BUILTINS: dict[str, dict[str, str]] = {
    "loco": {
        "null_loco": "common.cerberus_policies.backends.null_loco:NullLocoBackend",
        "scripted_loco": "common.cerberus_policies.backends.scripted_loco:ScriptedLocoBackend",
    },
    "arm": {},
    "body": {},
}

_BACKENDS: dict[str, dict[str, Factory]] = {slot: dict(v) for slot, v in _BUILTINS.items()}
_INSTANCES: dict[tuple[str, str], PolicyBackend] = {}


def _check_slot(slot: str) -> None:
    if slot not in SLOT_ENV:
        raise ValueError(f"Unknown policy slot {slot!r}. Slots: {sorted(SLOT_ENV)}")


def register_backend(slot: str, name: str, factory: Factory) -> None:
    """Register a backend factory (callable or lazy 'module:Class') under a slot."""
    _check_slot(slot)
    if name in _BACKENDS[slot]:
        raise ValueError(f"Policy backend {name!r} is already registered in slot {slot!r}.")
    _BACKENDS[slot][name] = factory


def available_backends(slot: str) -> list[str]:
    _check_slot(slot)
    return sorted(_BACKENDS[slot])


def _resolve(factory: Factory) -> Callable[..., PolicyBackend]:
    if isinstance(factory, str):
        module_name, _, attr = factory.partition(":")
        return getattr(importlib.import_module(module_name), attr)
    return factory


def get_backend(slot: str, name: str | None = None, config: dict[str, Any] | None = None) -> PolicyBackend:
    """Resolve explicit name, else the slot's env var. Instances are cached per (slot, name) unless
    ``config`` kwargs are passed to the backend constructor (then a fresh, uncached instance)."""
    _check_slot(slot)
    env = SLOT_ENV[slot]
    chosen = name or os.environ.get(env)
    if not chosen:
        raise ValueError(
            f"No {slot} policy backend selected: pass a name or set {env}. Available: {available_backends(slot)}"
        )
    if chosen not in _BACKENDS[slot]:
        raise ValueError(f"Unknown {slot} policy backend {chosen!r}. Available: {available_backends(slot)}")
    factory = _resolve(_BACKENDS[slot][chosen])
    if config is not None:
        return factory(**config)
    key = (slot, chosen)
    if key not in _INSTANCES:
        _INSTANCES[key] = factory()
    return _INSTANCES[key]


def _reset_for_tests() -> None:
    """Restore built-in backends only and drop cached instances."""
    _BACKENDS.clear()
    _BACKENDS.update({slot: dict(v) for slot, v in _BUILTINS.items()})
    _INSTANCES.clear()
