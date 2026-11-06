"""Simulation-launcher CLI helpers that work on both the old and the new Isaac Lab line.

Pure python at import time: ``isaaclab`` is only imported inside :func:`add_launcher_args`.

Why this exists -- the ``--headless`` question
----------------------------------------------
Isaac Lab 3.0's own documentation contradicts itself: a migration guide says ``--headless`` is
independent of ``--viz``, the release notes say it was removed. What the ``release/3.0.0`` source
(``isaaclab_physx.app.KitLauncher``) actually does:

* ``--headless`` is *not* registered by ``add_launcher_args`` any more, but the launcher still reads
  ``launcher_args.get("headless")`` and the ``HEADLESS`` environment variable.
* The viewport only opens when a Kit visualizer was requested (``--viz kit``; ``--viz`` is an alias of
  ``--visualizer``, argparse ``dest`` is ``visualizer``). Without it Kit is forced headless.
* Lab 2.3.2 / Sim 5.1.0 (``AppLauncher``): ``--headless`` and ``HEADLESS`` decide; there is no ``--viz``.

So neither flag may be assumed to exist. :func:`add_launcher_args` re-adds a ``--headless`` flag when the
installed Lab dropped it (so existing command lines and pod scripts keep parsing instead of leaking into
the Hydra overrides), and :func:`headless_requested` answers "will no Kit viewport open?" from whichever
of ``args.headless`` / ``args.viz`` (or ``args.visualizer``) / ``args.livestream`` / ``$HEADLESS`` exist.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping

_TRUE_STRINGS = frozenset({"1", "true", "yes", "on"})
_VIZ_ATTRS = ("viz", "visualizer")
"""Namespace attributes that may carry the ``--viz`` selection (Lab 3.0's argparse ``dest`` is ``visualizer``)."""


def add_launcher_args(parser: argparse.ArgumentParser) -> None:
    """Add the Isaac Sim launcher arguments to ``parser`` on either Isaac Lab line.

    Delegates to ``isaaclab.app.AppLauncher.add_app_launcher_args`` (Lab 2.3.2: the real launcher; Lab 3.0:
    a deprecated adapter around ``isaaclab.app.add_launcher_args``). If the installed Lab no longer
    registers ``--headless`` (Lab 3.0), a plain ``--headless`` store_true flag is added so it still parses;
    Lab 3.0's launcher honours ``args.headless`` when present.

    Must be called after the caller has made ``isaaclab`` importable, which is already true for the
    train/play scripts (they import ``AppLauncher`` before parsing).
    """
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    try:
        parser.add_argument(
            "--headless",
            action="store_true",
            default=False,
            help="Run without opening a Kit viewport (compat flag; Isaac Lab 3.0 dropped it from its own parser).",
        )
    except argparse.ArgumentError:
        pass  # AppLauncher (Lab 2.x) already registered --headless


def _get(args, name: str, default=None):
    """Read ``name`` from an ``argparse.Namespace``-like object or a mapping."""
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


def _has(args, name: str) -> bool:
    if isinstance(args, Mapping):
        return name in args
    return hasattr(args, name)


def _visualizer_names(value) -> list[str]:
    """Normalise a ``--viz`` value (None, 'none', 'kit,rerun', ['kit'], []) to lowercase names, 'none' dropped."""
    if value is None:
        return []
    items = value.split(",") if isinstance(value, str) else list(value)
    names = [str(item).strip().lower() for item in items]
    return [name for name in names if name and name != "none"]


def _env_flag(name: str, environ: Mapping[str, str]) -> bool:
    return str(environ.get(name, "")).strip().lower() in _TRUE_STRINGS


def headless_requested(args, environ: Mapping[str, str] | None = None) -> bool:
    """True when no Kit viewport window will open for this launch.

    Never assumes ``--headless`` or ``--viz`` exists. Headless is requested when any of these hold:

    1. the ``HEADLESS`` environment variable is truthy (``1``/``true``/``yes``/``on``);
    2. ``args.headless`` is truthy (Lab 2.x, or the compat flag from :func:`add_launcher_args`);
    3. ``args.livestream`` is 1 or 2 (livestreaming always runs headless on the host);
    4. a ``--viz`` selection exists (``args.viz`` or ``args.visualizer``; ``None``, ``'none'``, an empty
       list or any selection without ``kit`` all mean "no Kit window", matching Lab 3.0 forcing Kit headless
       unless ``--viz kit`` was passed).

    Otherwise False. On the old stack (no ``viz`` attribute) the result is just ``headless or $HEADLESS``,
    i.e. exactly what ``AppLauncher`` decides.

    Args:
        args: Parsed CLI namespace (or mapping), e.g. ``args_cli``.
        environ: Environment mapping; defaults to ``os.environ``.
    """
    environ = os.environ if environ is None else environ
    if _env_flag("HEADLESS", environ):
        return True
    if bool(_get(args, "headless", False)):
        return True
    if _get(args, "livestream", -1) in (1, 2):
        return True
    for attr in _VIZ_ATTRS:
        if _has(args, attr):
            return "kit" not in _visualizer_names(_get(args, attr))
    return False


def local_rank(app_launcher, environ: Mapping[str, str] | None = None) -> int:
    """Local GPU rank for distributed runs.

    Lab 2.x ``AppLauncher`` exposes ``.local_rank``; Lab 3.0's deprecated adapter does not, so fall back to
    the ``LOCAL_RANK`` environment variable that torchrun sets (which is what the old attribute was read from).
    """
    value = getattr(app_launcher, "local_rank", None)
    if value is not None:
        return int(value)
    environ = os.environ if environ is None else environ
    return int(environ.get("LOCAL_RANK", "0"))
