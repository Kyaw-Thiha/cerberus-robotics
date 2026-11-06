"""World backend registry. Backends are chosen explicitly (argument or the
WORLD_BACKEND env var) -- there is deliberately no silent default, so a run
never quietly lands on the wrong simulator stack.

Factories may be lazy "module:Class" strings so heavy backends are only
imported when first requested.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable

from common.cerberus_worlds.base import WorldBackend

WORLD_BACKEND_ENV = "WORLD_BACKEND"

_BUILTINS: dict[str, str] = {
    "arena": "common.cerberus_worlds.backends.arena_backend:ArenaBackend",
    "plain_usd": "common.cerberus_worlds.backends.plain_usd_backend:PlainUsdBackend",
}

_BACKENDS: dict[str, str | Callable[[], WorldBackend]] = dict(_BUILTINS)
_INSTANCES: dict[str, WorldBackend] = {}


def register_backend(name: str, factory: str | Callable[[], WorldBackend]) -> None:
    """Register a backend factory (callable or lazy 'module:Class')."""
    if name in _BACKENDS:
        raise ValueError(f"World backend {name!r} is already registered.")
    _BACKENDS[name] = factory


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def _instantiate(factory: str | Callable[[], WorldBackend]) -> WorldBackend:
    if isinstance(factory, str):
        module_name, _, attr = factory.partition(":")
        factory = getattr(importlib.import_module(module_name), attr)
    return factory()


def get_backend(name: str | None = None) -> WorldBackend:
    """Resolve explicit name, else $WORLD_BACKEND; instances are cached."""
    chosen = name or os.environ.get(WORLD_BACKEND_ENV)
    if not chosen:
        raise ValueError(
            f"No world backend selected: pass a name or set {WORLD_BACKEND_ENV}. "
            f"Available: {available_backends()}"
        )
    if chosen not in _BACKENDS:
        raise ValueError(f"Unknown world backend {chosen!r}. Available: {available_backends()}")
    if chosen not in _INSTANCES:
        _INSTANCES[chosen] = _instantiate(_BACKENDS[chosen])
    return _INSTANCES[chosen]


def _reset_for_tests() -> None:
    """Restore built-in backends only and drop cached instances."""
    _BACKENDS.clear()
    _BACKENDS.update(_BUILTINS)
    _INSTANCES.clear()
