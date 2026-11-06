"""Scene admission manifest: which scenes passed the physics/spawn/lidar checks, and where the robot may start.

Pure python (no isaaclab, no pxr). Written by ``scripts/validate_scenes.py`` (a pod script), read by anything
that picks scenes (SLAM eval, Phase 1 navigation).

File: ``<scenes_root>/<source>/admission.json`` (``scenes_root`` honours ``$CERBERUS_SCENES_DIR``)::

    {
      "schema_version": 1,
      "source": "grscenes",
      "generated_utc": "2026-10-06T12:00:00Z",
      "tool_versions": {"isaaclab_sha": "...", "arena_sha": "...", "isaacsim_image": "6.1.0"},
      "scenes": [ {AdmissionRow as dict}, ... ]
    }

Rows are keyed by ``scene_id`` (``"<source>:<folder name>"``, the id ``arena_assets.discover_scenes`` emits). Failed
scenes stay in the file with their reasons; consumers must only use ``status == "pass"`` rows.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from common.cerberus_worlds.arena_assets import scenes_root

SCHEMA_VERSION = 1
MANIFEST_NAME = "admission.json"
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"
VALID_STATUSES = (STATUS_PASS, STATUS_FAIL, STATUS_SKIP)


@dataclass(frozen=True)
class SpawnPose:
    """A validated robot start pose in the SCENE frame: ``(x, y)`` metres, ``yaw`` rad about +z, and the measured
    ``floor_z`` (metres) under that point. Spawn the robot base at ``floor_z + the robot's default standing height``."""

    x: float
    y: float
    yaw: float
    floor_z: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "yaw", "floor_z"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"SpawnPose.{name} must be finite, got {getattr(self, name)!r}")


@dataclass(frozen=True)
class AdmissionRow:
    scene_id: str
    source: str
    category: str = ""
    backend: str = ""  # "plain_usd" | "arena"
    status: str = STATUS_SKIP
    reasons: tuple[str, ...] = ()
    spawn_poses: tuple[SpawnPose, ...] = ()  # best first; non-empty for every "pass" row
    shared_scene: bool = True  # scene cloning mode the pass was measured with
    metrics: dict[str, Any] = field(default_factory=dict)  # step_ms, peak_gpu_mib, build_s, tilt_deg, lidar_hits, ...
    date_utc: str = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}, got {self.status!r}")
        if self.status == STATUS_PASS and not self.spawn_poses:
            raise ValueError(f"{self.scene_id}: a 'pass' row needs at least one spawn pose")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        d["spawn_poses"] = [asdict(p) for p in self.spawn_poses]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AdmissionRow:
        d = dict(d)
        d["reasons"] = tuple(d.get("reasons", ()))
        d["spawn_poses"] = tuple(SpawnPose(**p) for p in d.get("spawn_poses", ()))
        return cls(**d)


def manifest_path(source: str, root: Path | str | None = None) -> Path:
    base = Path(root) if root is not None else scenes_root()
    return base / source / MANIFEST_NAME


def write_manifest(
    source: str,
    rows: list[AdmissionRow],
    *,
    tool_versions: dict[str, Any] | None = None,
    generated_utc: str = "",
    root: Path | str | None = None,
) -> Path:
    """Write (replace) the manifest atomically; rows are sorted by scene id. Returns the path."""
    ids = [r.scene_id for r in rows]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate scene ids in manifest: {dupes}")
    path = manifest_path(source, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "generated_utc": generated_utc,
        "tool_versions": dict(tool_versions or {}),
        "scenes": [r.to_dict() for r in sorted(rows, key=lambda r: r.scene_id)],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def load_manifest(source: str, root: Path | str | None = None) -> list[AdmissionRow]:
    """All rows (any status). Raises FileNotFoundError if no manifest exists, ValueError on a wrong schema."""
    path = manifest_path(source, root)
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {payload.get('schema_version')!r} != {SCHEMA_VERSION}")
    return [AdmissionRow.from_dict(r) for r in payload["scenes"]]


def admitted_scene_ids(source: str, category: str | None = None, root: Path | str | None = None) -> list[str]:
    """Scene ids with ``status == "pass"`` (optionally one category), sorted."""
    return sorted(
        r.scene_id
        for r in load_manifest(source, root)
        if r.status == STATUS_PASS and (category is None or r.category == category)
    )


def write_scene_ids_file(path: Path | str, scene_ids: Iterable[str]) -> int:
    """Write one scene id per line and return how many were written. An EMPTY list writes an EMPTY file.

    Callers must branch on the returned count, not on the file size: ``print("\\n".join([]))`` writes a lone
    newline, so a 1-byte file passes a size/`-s` test and is then handed to a consumer that reads zero ids and
    treats that as misuse (see ``scripts/run_scene_admission_pod.sh``'s upload step).
    """
    ids = [str(s) for s in scene_ids]
    Path(path).write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    return len(ids)


def spawn_pose(source: str, scene_id: str, index: int = 0, root: Path | str | None = None) -> SpawnPose:
    """The `index`-th validated spawn pose of an ADMITTED scene. Raises KeyError if the scene is not in the manifest
    and ValueError if it did not pass (never silently falls back to (0, 0, 0))."""
    for row in load_manifest(source, root):
        if row.scene_id == scene_id:
            if row.status != STATUS_PASS:
                raise ValueError(f"{scene_id} is not admitted (status={row.status}, reasons={list(row.reasons)})")
            return row.spawn_poses[index]
    raise KeyError(f"{scene_id} not in {manifest_path(source, root)}")
