"""Pure-python argument resolution for slam/slam/ground_truth/sim_ground_truth_loop.py (no isaaclab, no torch).

Kept separate so everything that must happen BEFORE the Isaac app launches (scene lookup, the MDL_SYSTEM_PATH
side effect, spawn lookup, checkpoint dispatch, the report) is unit-testable and shared by the
loop's ``--dry-run``.

Scenes are only ever reached through ``common.cerberus_worlds.registry.get_backend("plain_usd")`` and spawn poses
only through ``common.cerberus_worlds.admission.spawn_pose``. A scene never falls back to a (0, 0, 0) spawn.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common.cerberus_worlds import registry
from common.cerberus_worlds.admission import SpawnPose, spawn_pose

# gym task ids of the three SLAM-eval env cfgs (registered by sim_env_cfg.register_slam_eval_tasks(); kept here
# because choosing one must not need isaaclab).
TASK_ID_LIDAR_CAMERA = "Cerberus-Go2-SlamEval-LidarCamera-v0"
TASK_ID_CAMERA_ONLY = "Cerberus-Go2-SlamEval-CameraOnly-v0"
TASK_ID_CAMERA_STEREO = "Cerberus-Go2-SlamEval-CameraStereo-v0"

PLAIN_USD_BACKEND = "plain_usd"
SCENE_STOCK = "stock"
CHECKPOINT_KINDS = {".pt": "pt", ".onnx": "onnx"}
REPORT_NAME = "sim_loop_report.json"
DEFAULT_REPORT = "data/reports/slam_sim_loop.json"

_REPO_ROOT = Path(__file__).resolve().parents[3]


class RunArgError(ValueError):
    """A bad or unresolvable command-line combination (the loop exits 2)."""


@dataclass(frozen=True)
class ResolvedScene:
    scene_id: str  # "stock" or "<source>:<folder>"
    scene: dict[str, Any] | None  # plain_usd load_scene() dict; None for stock
    spawn: SpawnPose | None  # scene frame; None for stock
    spawn_source: str  # "stock" | "override" | "manifest:<source>[<index>]"

    def to_metadata(self) -> dict[str, Any]:
        scene = self.scene or {}
        return {
            "scene_id": self.scene_id,
            "usd_path": scene.get("usd_path"),
            "variant": scene.get("variant"),
            "mdl_path": scene.get("mdl_path"),
            "spawn": asdict(self.spawn) if self.spawn else None,
            "spawn_source": self.spawn_source,
        }


def task_id_for_sensor_rig(sensor_rig: str) -> str:
    """`sensor_rig` matches record_mission_bag.py's own --sensor-rig choices
    (`lidar_camera` / `camera_only` / `camera_stereo`), so both scripts agree
    on rig naming without importing each other."""
    return {
        "lidar_camera": TASK_ID_LIDAR_CAMERA,
        "camera_only": TASK_ID_CAMERA_ONLY,
        "camera_stereo": TASK_ID_CAMERA_STEREO,
    }[sensor_rig]


def parse_spawn_override(text: str) -> SpawnPose:
    """``"x,y,yaw,floor_z"`` (metres, metres, radians, metres; scene frame) -> SpawnPose."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise RunArgError(f"--spawn expects x,y,yaw,floor_z (4 comma-separated numbers), got {text!r}")
    try:
        x, y, yaw, floor_z = (float(p) for p in parts)
        return SpawnPose(x=x, y=y, yaw=yaw, floor_z=floor_z)
    except ValueError as e:
        raise RunArgError(f"--spawn {text!r}: {e}") from e


def checkpoint_kind(path: str) -> str:
    """``"pt"`` (rsl-rl runner checkpoint) or ``"onnx"`` (exported policy), chosen by file extension."""
    suffix = Path(path).suffix.lower()
    try:
        return CHECKPOINT_KINDS[suffix]
    except KeyError:
        raise RunArgError(f"--checkpoint must end in {sorted(CHECKPOINT_KINDS)}, got {path!r}") from None


def resolve_scene(scene_arg: str, spawn_index: int | None, spawn_override: str | None) -> ResolvedScene:
    """Resolve ``--scene`` / ``--spawn-index`` / ``--spawn``.

    For a non-stock scene this also sets ``MDL_SYSTEM_PATH`` (``prepare_scene_env``), so call it before the Isaac
    app launches. Raises :class:`RunArgError` if the scene is not on disk, or no spawn can be resolved.
    """
    if scene_arg == SCENE_STOCK:
        if spawn_index is not None or spawn_override is not None:
            raise RunArgError("--spawn-index/--spawn only apply to a scene; --scene stock spawns at the env origin")
        return ResolvedScene(SCENE_STOCK, None, None, "stock")

    if spawn_index is not None and spawn_override is not None:
        raise RunArgError("--spawn-index and --spawn are mutually exclusive")
    if spawn_index is not None and spawn_index < 0:
        raise RunArgError(f"--spawn-index must be >= 0, got {spawn_index}")

    backend = registry.get_backend(PLAIN_USD_BACKEND)
    try:
        scene_id = backend.normalize_scene_ref(scene_arg)
    except ValueError as e:
        raise RunArgError(str(e)) from e
    backend.prepare_scene_env(scene_id)  # MDL_SYSTEM_PATH: Kit reads it at startup
    try:
        scene = backend.load_scene(scene_id)
    except FileNotFoundError as e:
        raise RunArgError(str(e)) from e

    if spawn_override is not None:
        return ResolvedScene(scene_id, scene, parse_spawn_override(spawn_override), "override")

    index = 0 if spawn_index is None else spawn_index
    source = scene_id.partition(":")[0]
    hint = "run scripts/validate_scenes.py for this source, or pass --spawn x,y,yaw,floor_z for a smoke test"
    try:
        spawn = spawn_pose(source, scene_id, index)
    except FileNotFoundError as e:
        raise RunArgError(f"no admission manifest for source {source!r} ({e}); {hint}") from e
    except KeyError as e:
        raise RunArgError(f"{e.args[0]}; {hint}") from e
    except ValueError as e:
        raise RunArgError(f"{e}; pick an admitted scene or pass --spawn") from e
    except IndexError:
        raise RunArgError(f"{scene_id} has no spawn pose at --spawn-index {index}") from None
    return ResolvedScene(scene_id, scene, spawn, f"manifest:{source}[{index}]")


def report_path(backend: str | None, run_timestamp: str | None, override: str | None) -> Path:
    """``--report``, else beside the bag in the run dir record_mission_bag.py/sync_to_r2.py use, else data/reports."""
    if override:
        path = Path(override)
        return path if path.is_absolute() else _REPO_ROOT / path
    if backend and run_timestamp:
        return _REPO_ROOT / "slam/slam/eval/results" / backend / run_timestamp / REPORT_NAME
    return _REPO_ROOT / DEFAULT_REPORT


def write_report(path: Path, report: dict[str, Any]) -> bool:
    """Best-effort JSON write (False on failure, so the caller can turn a pass into a fail)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    except OSError as e:
        print(f"[sim_ground_truth_loop] ERROR: could not write report {path}: {e}", flush=True)
        return False
    return True
