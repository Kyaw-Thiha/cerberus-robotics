"""Refuse to fetch, preprocess or read scene assets anywhere but a pod.

Why: a GRScenes home pulls in tens of GB of shared `Materials`/`models` (H2 measured ~25 GB for the first
scene alone), and the development laptop has only a few GB free. A failed or partial fetch fills the disk and
leaves a half-written scene that later runs silently pick up. The owner's rule (2026-11-09) is therefore: scene
assets are downloaded, preprocessed and read on the pod only, and nothing scene-sized is ever downloaded to the
laptop. `docs/worlds.md` and `plans/handoff_h4_navigation_in_scene.md` state the same rule.

Where the guard sits: the CLI entry of every scene-touching script. Three of them (`fetch_grscenes_from_r2.py`,
`fetch_grscenes_to_r2.py`, `build_lidar_proxy.py`) are driven by existing tests through `main()` with mocked I/O,
so their guard is in the `__main__` block -- which is also the only documented way to run them. The rest guard
inside `main()` right after argument parsing, so `--help` still works.

Set `CERBERUS_ON_POD=1` on the pod. The pod driver scripts do this themselves.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

POD_ENV_VAR = "CERBERUS_ON_POD"
"""Exactly ``"1"`` means pod. ``"0"``, ``"true"`` and unset all mean the laptop."""

DEFAULT_REASON = (
    "scene assets are tens of GB and must be downloaded, preprocessed and read on the pod only; a local fetch "
    "fills the disk and leaves a half-written scene behind"
)


def is_pod(env: Mapping[str, str] | None = None) -> bool:
    """True only when ``CERBERUS_ON_POD`` is exactly ``"1"``."""
    return (os.environ if env is None else env).get(POD_ENV_VAR) == "1"


def require_pod(script: str, *, reason: str | None = None) -> None:
    """Exit with a clear error unless this process is on a pod."""
    if is_pod():
        return
    raise SystemExit(
        f"{script} is pod-only: {reason or DEFAULT_REASON}.\n"
        f"Set {POD_ENV_VAR}=1 on the pod and run it there (see docs/worlds.md). Nothing was downloaded or written."
    )
