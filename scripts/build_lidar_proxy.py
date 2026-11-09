"""Build the LiDAR proxy mesh of a GRScenes scene: ``<scene folder>/lidar_proxy.usd`` (+ ``lidar_proxy.json``).

Why: Isaac Lab's MultiMeshRayCaster cannot ray-cast against a raw GRScenes house, because over a third of its Mesh
prims are empty (``points`` / ``faceVertexCounts`` return None) and the sensor raises ``TypeError: iteration over a 0-d
array``. The proxy is ONE static triangle ``Mesh`` (``/Proxy``) holding only the valid geometry in world frame, no
physics/collision/rigid-body APIs. slam/slam/eval/scene_env_cfg.apply_scene spawns it, invisible to cameras, at the
world origin (where the scene USD itself is spawned) and points the LiDAR at it.

Needs only ``pxr`` (no Kit app): on a pod ``/isaac-sim/python.sh scripts/build_lidar_proxy.py --scene grscenes:<id>``
(the SLAM eval script does this on its own when the proxy is missing); locally ``usd-core`` works. Meters and Z-up are
required (the sim spawns the scene USD unconverted, so converting here would put the proxy somewhere else); other
stages are refused.

Exit codes: 0 built or cached, 1 failure, 2 bad args. ``--dry-run`` only resolves the paths (works without pxr).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.cerberus_worlds import registry
from common.cerberus_worlds.scene_geometry import (
    PROXY_MESH_PRIM,
    PROXY_ROOT_PRIM,
    PROXY_SIDECAR_NAME,
    gather_world_triangles,
    lidar_proxy_path,
    write_triangle_mesh_usd,
)

PLAIN_USD_BACKEND = "plain_usd"
DEFAULT_BUDGET_S = 1800.0
SCHEMA_VERSION = 3  # 3: Xform root with the mesh below it (see scene_geometry.PROXY_ROOT_PRIM)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="grscenes:<folder> (resolved through the plain_usd backend)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the cached proxy is up to date")
    parser.add_argument("--budget_s", type=float, default=DEFAULT_BUDGET_S, help="abort gathering after this many seconds")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="print the resolved paths and cache state; import no pxr")
    return parser


def sidecar_path(proxy: Path) -> Path:
    return proxy.with_name(PROXY_SIDECAR_NAME)


def is_fresh(source: Path, proxy: Path) -> bool:
    """The cached proxy was built from this exact version of the source USD (not just a newer file)."""
    sidecar = sidecar_path(proxy)
    if not (proxy.is_file() and sidecar.is_file()):
        return False
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return False
    return (
        meta.get("schema_version") == SCHEMA_VERSION
        and meta.get("source_mtime_ns") == source.stat().st_mtime_ns
        and proxy.stat().st_mtime_ns >= source.stat().st_mtime_ns
    )


def check_stage_units(stage: Any) -> dict[str, Any]:
    """Read the source stage's units and require only Z-up.

    GRScenes houses are authored in CENTIMETRES (``meters_per_unit = 0.01``); Isaac consumes them as a reference
    and scales by that factor, so the proxy must declare the SAME units and be written in the source's own
    coordinates, not converted here (a metre proxy would be 100x too small next to the scene).
    """
    from pxr import UsdGeom

    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if up_axis != "Z":
        raise ValueError(f"scene stage is up_axis={up_axis!r}; the proxy builder only handles Z-up scenes")
    return {"up_axis": up_axis, "meters_per_unit": meters_per_unit}


def write_proxy_usd(points, triangles, out: Path, units: dict[str, Any]) -> None:
    """One static triangle Mesh at ``/Proxy`` (default prim), nothing else: no physics, collision or material.

    The writer itself lives in ``scene_geometry`` so the height-scan proxy (``scripts/build_scan_proxy.py``) and this
    one cannot drift apart.
    """
    write_triangle_mesh_usd(
        points,
        triangles,
        out,
        meters_per_unit=float(units["meters_per_unit"]),
        root_prim=PROXY_ROOT_PRIM,
        mesh_prim=PROXY_MESH_PRIM,
    )


def build_proxy(source: Path, proxy: Path, budget_s: float) -> dict[str, Any]:
    """Gather the valid world-frame triangles of ``source`` and write ``proxy`` + sidecar; returns the sidecar dict."""
    from pxr import Usd

    t0 = time.perf_counter()
    stage = Usd.Stage.Open(str(source))
    if not stage:
        raise RuntimeError(f"cannot open {source}")
    units = check_stage_units(stage)
    stats: dict[str, int] = {}
    points, triangles = gather_world_triangles(stage, "/", frozenset(), budget_s, stats)
    write_proxy_usd(points, triangles, proxy, units)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "source_mtime_ns": source.stat().st_mtime_ns,
        "vertices": len(points),
        "triangles": len(triangles),
        "bounds_min": [round(float(v), 3) for v in points.min(axis=0)],
        "bounds_max": [round(float(v), 3) for v in points.max(axis=0)],
        "build_s": round(time.perf_counter() - t0, 1),
        **stats,
        **units,
    }
    sidecar_path(proxy).write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        backend = registry.get_backend(PLAIN_USD_BACKEND)
        info = backend.describe_scene(backend.normalize_scene_ref(args.scene))
    except ValueError as e:
        print(f"[build_lidar_proxy] ERROR: {e}", file=sys.stderr)
        return 2
    if not info.get("usd_path"):
        print(f"[build_lidar_proxy] ERROR: {info.get('error') or 'scene has no USD (is it stock?)'}", file=sys.stderr)
        return 2
    source = Path(info["usd_path"])
    proxy = lidar_proxy_path(source)
    fresh = not args.force and is_fresh(source, proxy)
    print(f"[build_lidar_proxy] scene={args.scene} source={source} proxy={proxy} cached={fresh}", flush=True)
    if args.dry_run:
        return 0
    if fresh:
        print(f"[build_lidar_proxy] up to date: {sidecar_path(proxy).read_text()}")
        return 0
    try:
        meta = build_proxy(source, proxy, args.budget_s)
    except Exception as e:  # noqa: BLE001 - loud, nonzero, with the cause
        traceback.print_exc()
        print(f"[build_lidar_proxy] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"[build_lidar_proxy] wrote {proxy}: {json.dumps(meta, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
