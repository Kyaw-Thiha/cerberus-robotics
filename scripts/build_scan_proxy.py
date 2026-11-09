#!/usr/bin/env python3
"""Build the height-scan proxy mesh of a GRScenes scene: ``<scene folder>/scan_proxy.usd`` (+ sidecar).

Why: the locomotion policy's 187-cell height scanner casts straight DOWN from 20 m above the base
(``contracts/heightscan_ppo_v1.yaml``). Against the raw house it hits the ceiling first, so every cell would read the
same value and obstacles would be invisible; and Isaac Lab's multi-mesh ray caster cannot parse the raw house at all
(over a third of its Mesh prims are empty -- see ``scripts/build_lidar_proxy.py``). This proxy is ONE static triangle
mesh of the valid world-frame geometry in METRES, with large high horizontal slabs (ceilings/roofs) removed and
everything else -- floors, walls including their top faces, furniture -- kept. See
``common.cerberus_worlds.scene_geometry.filter_scan_triangles`` for the rule and its limits.

Needs only ``pxr`` (no Kit app): on a pod ``/isaac-sim/python.sh scripts/build_scan_proxy.py --scene grscenes:<id>``;
locally ``usd-core`` works. Exit codes: 0 built or cached, 1 failure, 2 bad args. ``--dry-run`` resolves the paths
only (works without pxr).

Usage:
    python3 scripts/build_scan_proxy.py --scene grscenes:<id> --floor-z 0.0 [--cutoff 0.6] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.cerberus_worlds import registry
from common.cerberus_worlds.scene_geometry import (
    SCAN_PROXY_NAME,
    SCAN_PROXY_SIDECAR_NAME,
    build_scan_proxy,
    scan_proxy_path,
)

PLAIN_USD_BACKEND = "plain_usd"
DEFAULT_BUDGET_S = 1800.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="grscenes:<folder> (resolved through the plain_usd backend)")
    parser.add_argument("--floor-z", type=float, default=None,
                        help="main floor height in the scene frame, metres (default: the occupancy grid's, if built)")
    parser.add_argument("--cutoff", type=float, default=0.6, help="unused by the filter, recorded for the record")
    parser.add_argument("--ceiling-min-area", type=float, default=3.0, help="m^2; bigger high horizontal slabs are ceilings")
    parser.add_argument("--ceiling-min-height", type=float, default=1.5, help="m above the floor before a slab can be a ceiling")
    parser.add_argument("--budget_s", type=float, default=DEFAULT_BUDGET_S, help="abort gathering after this many seconds")
    parser.add_argument("--force", action="store_true", help="rebuild even if the cached proxy is up to date")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="print resolved paths; import no pxr")
    return parser


def occupancy_floor_z(scene_usd: Path) -> float | None:
    """``floor_z`` recorded by ``planning/occupancy.py``'s grid for this scene, if it has been built."""
    grid = scene_usd.parent / "occupancy.npz"
    if not grid.is_file():
        return None
    try:
        import numpy as np

        with np.load(grid, allow_pickle=False) as data:
            return float(data["floor_z"]) if "floor_z" in data else None
    except Exception:  # noqa: BLE001 - a broken grid is a finding, not a reason to crash the builder
        return None


def is_fresh(source: Path, proxy: Path, params: dict[str, float]) -> bool:
    sidecar = proxy.with_name(SCAN_PROXY_SIDECAR_NAME)
    if not (proxy.is_file() and sidecar.is_file()):
        return False
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return False
    return (
        meta.get("source_mtime_ns") == source.stat().st_mtime_ns
        and proxy.stat().st_mtime_ns >= source.stat().st_mtime_ns
        and all(meta.get(key) == value for key, value in params.items())
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from pod_guard import require_pod

    require_pod("build_scan_proxy.py")
    try:
        backend = registry.get_backend(PLAIN_USD_BACKEND)
        info = backend.describe_scene(backend.normalize_scene_ref(args.scene))
    except ValueError as e:
        print(f"[build_scan_proxy] ERROR: {e}", file=sys.stderr)
        return 2
    if not info.get("usd_path"):
        print(f"[build_scan_proxy] ERROR: {info.get('error') or 'scene has no USD (is it stock?)'}", file=sys.stderr)
        return 2
    source = Path(info["usd_path"])
    proxy = scan_proxy_path(source)
    floor_z = args.floor_z
    if floor_z is None:
        floor_z = occupancy_floor_z(source)
    if floor_z is None:
        print("[build_scan_proxy] ERROR: no --floor-z and no occupancy grid to read it from", file=sys.stderr)
        return 2
    params = {
        "floor_z": float(floor_z),
        "cutoff_m": float(args.cutoff),
        "ceiling_min_area_m2": float(args.ceiling_min_area),
        "ceiling_min_height_m": float(args.ceiling_min_height),
    }
    fresh = not args.force and is_fresh(source, proxy, params)
    print(f"[build_scan_proxy] scene={args.scene} source={source} proxy={proxy} cached={fresh}", flush=True)
    if args.dry_run:
        return 0
    if fresh:
        print(f"[build_scan_proxy] up to date: {proxy.with_name(SCAN_PROXY_SIDECAR_NAME).read_text()}")
        return 0
    try:
        meta = build_scan_proxy(source, proxy, budget_s=args.budget_s, **params)
    except Exception as e:  # noqa: BLE001 - loud, nonzero, with the cause
        traceback.print_exc()
        print(f"[build_scan_proxy] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"[build_scan_proxy] wrote {proxy} ({SCAN_PROXY_NAME}): {json.dumps(meta, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
