"""Out-of-tree Isaac Lab Arena assets for Cerberus, plus pure-python scene-data helpers.

Everything importable from this package works with plain python (no isaaclab, no pxr, no Arena): the
Arena/Isaac imports live inside functions. ``go2_embodiment`` holds the Arena embodiment, the GRScenes
background and the waypoint task; this file holds the on-disk scene layout helpers that both world
backends (``arena_backend``, ``plain_usd_backend``) share.

On-disk GRScenes layout (written by ``scripts/fetch_grscenes_from_r2.py``; gitignored data, CC BY-NC-SA 4.0)::

    data/scenes/grscenes/home/Materials/            mdl + textures      -> MDL_SYSTEM_PATH
    data/scenes/grscenes/home/models/               layout + object usd
    data/scenes/grscenes/home/scenes/<id>_usd/      start_result_{raw,navigation,interaction}.usd, metadata.json

``CERBERUS_SCENES_DIR`` overrides the ``data/scenes`` root. A scene is any folder (up to ``MAX_SCAN_DEPTH``
levels below ``data/scenes/<source>/``) that contains ``start_result_navigation.usd`` or
``start_result_raw.usd``; the navigation variant is preferred. Scene id = ``<source>:<folder name>``.

How Arena is used from an external repo (read from ``third_party/IsaacLab-Arena``, fetched by
``scripts/fetch_arena.py``; see the docstring of ``go2_embodiment`` for the API names):
``docs/pages/arena_in_your_repo/external_tasks_and_embodiments.rst``.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SCENES_DIR_ENV = "CERBERUS_SCENES_DIR"
MDL_SYSTEM_PATH_ENV = "MDL_SYSTEM_PATH"
NAV_USD = "start_result_navigation.usd"
RAW_USD = "start_result_raw.usd"
SCENE_USD_NAMES: tuple[str, ...] = (NAV_USD, RAW_USD)  # preference order
DEFAULT_SOURCE = "grscenes"
MAX_SCAN_DEPTH = (
    4  # data/scenes/<source>/home/scenes/<id>_usd is depth 3 below <source>
)


def repo_root() -> Path:
    """Repo root (this file is <root>/common/cerberus_worlds/arena_assets/__init__.py)."""
    return Path(__file__).resolve().parents[3]


def scenes_root() -> Path:
    """The scenes data root: $CERBERUS_SCENES_DIR if set, else <repo>/data/scenes."""
    override = os.environ.get(SCENES_DIR_ENV)
    return Path(override).expanduser() if override else repo_root() / "data" / "scenes"


@dataclass(frozen=True)
class SceneFiles:
    """One plain-USD scene folder found on disk."""

    scene_id: str  # "<source>:<folder name>"
    source: str
    category: str  # "home" for .../home/scenes/<id>, else ""
    folder: Path
    usd_path: Path
    variant: str  # "navigation" | "raw"
    mdl_path: Path | None  # the Materials dir MDL_SYSTEM_PATH should point to, if found

    def to_metadata(self) -> dict[str, str | None]:
        return {
            "folder": str(self.folder),
            "usd_path": str(self.usd_path),
            "variant": self.variant,
            "mdl_path": str(self.mdl_path) if self.mdl_path else None,
        }


def scene_usd_in(folder: Path) -> Path | None:
    """The scene USD in `folder`, preferring start_result_navigation.usd over start_result_raw.usd."""
    for name in SCENE_USD_NAMES:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def mdl_path_for(folder: Path) -> Path | None:
    """Find the ``Materials`` dir a scene folder refers to.

    Scene folders hold a ``Materials`` symlink (recreated by fetch_grscenes_from_r2.py) to
    ``../../Materials``; walk up a few levels so a tree without the link still resolves.
    """
    for candidate in (folder, *list(folder.parents)[:3]):
        materials = candidate / "Materials"
        if materials.is_dir():
            return materials
    return None


def sim_usd_path(files: SceneFiles) -> Path:
    """The USD the SIMULATION should load for `files`, in metres (see ``scene_geometry.metric_scene_usd``).

    GRScenes are centimetre-authored and Isaac does not convert units, so the raw scene USD is spawned 100x too big.
    This returns a cached metre-scale wrapper beside it.

    It must NOT import ``pxr`` here: callers run this before the Isaac app launches, and importing ``pxr`` first breaks
    ``omni.usd`` (``AttributeError: module 'omni.usd' has no attribute 'get_context'`` at launch, seen on a pod). So the
    answer comes from a JSON sidecar when it is up to date, and otherwise from a child interpreter running
    ``python -m common.cerberus_worlds.scene_geometry``. If that child has no ``pxr`` (a local machine with neither
    Isaac Sim nor usd-core, where nothing can be spawned anyway) the original path is returned. A scene USD that cannot
    be opened raises ``RuntimeError`` instead of silently falling back to the wrong scale.
    """
    from common.cerberus_worlds import scene_geometry

    cached = scene_geometry.cached_metric_scene_usd(files.usd_path)
    if cached is not None:
        return cached
    proc = subprocess.run(
        [sys.executable, "-m", "common.cerberus_worlds.scene_geometry", str(files.usd_path)],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == scene_geometry.PXR_MISSING_EXIT:
        return files.usd_path
    if proc.returncode != 0:
        raise RuntimeError(f"cannot prepare the metre-scale scene for {files.usd_path}: {proc.stderr.strip()[-400:]}")
    return Path(proc.stdout.strip().splitlines()[-1])


def find_scene_dirs(base: Path, max_depth: int = MAX_SCAN_DEPTH) -> list[Path]:
    """Folders under `base` (depth-limited, sorted) that directly contain a scene USD."""
    base = Path(base)
    if not base.is_dir():
        return []
    found: list[Path] = []
    stack: list[tuple[Path, int]] = [(base, 0)]
    while stack:
        current, depth = stack.pop()
        if scene_usd_in(current) is not None:
            found.append(current)
            continue  # a scene folder is a leaf; do not descend into its asset subfolders
        if depth >= max_depth:
            continue
        try:
            children = [p for p in current.iterdir() if p.is_dir()]
        except OSError:
            continue
        stack.extend((child, depth + 1) for child in children)
    return sorted(found)


def discover_scenes(
    source: str = DEFAULT_SOURCE, root: Path | None = None
) -> list[SceneFiles]:
    """Scan ``<root or scenes_root()>/<source>/`` for scene folders (sorted by scene id)."""
    base = (Path(root) if root is not None else scenes_root()) / source
    scenes = []
    for folder in find_scene_dirs(base):
        usd = scene_usd_in(folder)
        assert usd is not None
        scenes.append(
            SceneFiles(
                scene_id=f"{source}:{folder.name}",
                source=source,
                category=folder.parent.parent.name
                if folder.parent.name == "scenes"
                else "",
                folder=folder,
                usd_path=usd,
                variant="navigation" if usd.name == NAV_USD else "raw",
                mdl_path=mdl_path_for(folder),
            )
        )
    return sorted(scenes, key=lambda s: s.scene_id)


def find_scene(
    scene_id: str, source: str = DEFAULT_SOURCE, root: Path | None = None
) -> SceneFiles:
    """Look up ``<source>:<folder>`` (a bare ``<folder>`` is accepted too) or raise with what is available."""
    prefix = f"{source}:"
    wanted = scene_id if scene_id.startswith(prefix) else prefix + scene_id
    scenes = discover_scenes(source, root)
    for scene in scenes:
        if scene.scene_id == wanted:
            return scene
    base = (Path(root) if root is not None else scenes_root()) / source
    available = ", ".join(s.scene_id for s in scenes[:10]) or "none"
    raise FileNotFoundError(
        f"Scene {wanted!r} not found under {base} (available: {available}). "
        f"Fetch GRScenes with `python3 scripts/fetch_grscenes_from_r2.py` or set {SCENES_DIR_ENV}."
    )


def join_mdl_system_path(existing: str | None, new: str) -> str:
    """Append `new` to a ``:``-separated MDL_SYSTEM_PATH value unless it is already present."""
    parts = [p for p in (existing or "").split(os.pathsep) if p]
    if new not in parts:
        parts.append(new)
    return os.pathsep.join(parts)


UE4_MDL_MODULES: tuple[str, ...] = ("OmniUe4Function.mdl", "OmniUe4Translucent.mdl", "OmniUe4Base.mdl")
"""The Omniverse UE4 helper modules GRScenes' own materials import (``using .::OmniUe4Function import *;`` and the
Translucent/Base twins). The dataset does not ship them and the import is RELATIVE (``.::``), so they must sit in the
same directory as the material ``.mdl`` files; otherwise every material fails to compile and renders flat red."""

UE4_MDL_SOURCE_SUBPATH = Path("kit") / "mdl" / "core" / "Ue4"
"""Where Isaac Sim ships those modules, relative to the Isaac Sim root."""


def isaac_sim_root() -> Path:
    return Path(os.environ.get("ISAACSIM_ROOT_PATH", "/isaac-sim"))


def install_ue4_mdl_modules(mdl_dir: Path | str, source_dir: Path | str | None = None) -> list[str]:
    """Copy the UE4 MDL helper modules from Isaac Sim into `mdl_dir` (the scene's ``Materials/`` folder).

    Returns the file names copied (empty when everything is already there). Existing files are never overwritten.
    Raises FileNotFoundError if Isaac Sim's copy of a module is missing; there is nothing to copy on a machine
    without Isaac Sim, so callers that may run there must guard on that themselves.

    Evidence (2026-10-06 pod run): the scene's ``Materials/*.mdl`` all start with ``using .::OmniUe4Function import
    *;``, the dataset has no ``OmniUe4*.mdl``, the runtime logged ``EntityResolver FAILED .::OmniUe4Function`` for
    every material, and the camera frame was uniformly red.
    """
    src = Path(source_dir) if source_dir is not None else isaac_sim_root() / UE4_MDL_SOURCE_SUBPATH
    dest = Path(mdl_dir)
    copied: list[str] = []
    for name in UE4_MDL_MODULES:
        target = dest / name
        if target.exists():
            continue
        source = src / name
        if not source.is_file():
            raise FileNotFoundError(f"Isaac Sim's {name} not found at {source}")
        shutil.copyfile(source, target)
        copied.append(name)
    return copied


def ensure_mdl_system_path(mdl_path: Path | str | None) -> bool:
    """Make ``$MDL_SYSTEM_PATH`` include `mdl_path` and install the UE4 MDL modules there. Returns True if the
    environment variable changed.

    GRScenes' MDL materials only resolve if the variable is set AND the UE4 helper modules sit beside them
    (:func:`install_ue4_mdl_modules`). Kit reads the variable at startup, so call this BEFORE the SimulationApp /
    AppLauncher is created (``scripts/arena_spike_go2.py`` does). When Isaac Sim is not installed (local dev) the
    module copy is skipped.
    # UNVERIFIED: that setting it after app launch is too late for Kit's MDL search path (assumed).
    # UNVERIFIED: that installing the modules turns the red materials into textured ones (a render run is needed).
    """
    if not mdl_path:
        return False
    if (isaac_sim_root() / UE4_MDL_SOURCE_SUBPATH).is_dir():
        install_ue4_mdl_modules(mdl_path)
    old = os.environ.get(MDL_SYSTEM_PATH_ENV)
    new = join_mdl_system_path(old, str(mdl_path))
    if new == (old or ""):
        return False
    os.environ[MDL_SYSTEM_PATH_ENV] = new
    return True


def spawn_pose_to_background_pose(
    spawn_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Pose to give the scene USD so the robot (at the env origin, yaw 0) starts at `spawn_pose` inside it.

    `spawn_pose` is ``(x, y, yaw_rad)`` in the scene USD frame (z-up, floor at z=0 assumed). The returned
    ``(position_xyz, quaternion_xyzw)`` is the scene's pose in the env frame: the inverse of the spawn
    transform, i.e. rotation Rz(-yaw) and translation -Rz(-yaw) @ (x, y, 0).
    """
    x, y, yaw = (float(v) for v in spawn_pose)
    c, s = math.cos(-yaw), math.sin(-yaw)
    position = (-(c * x - s * y), -(s * x + c * y), 0.0)
    half = -yaw / 2.0
    quat_xyzw = (0.0, 0.0, math.sin(half), math.cos(half))
    return position, quat_xyzw
