#!/usr/bin/env python3
"""T4 numbers for JEPLO: the same seeded start-goal pairs, planner and controller as the PPO baseline.

    /isaac-sim/python.sh scripts/jeplo_nav_eval.py --headless --scene grscenes:<id> --seed 0 --pairs 10 \
        --models third_party/jeplo/training/logs/rsl_rl/go2_loco/pretrained/exported --out outputs/scratch/jeplo/nav_eval

Writes one JSON record per pair plus summary.json. Ground-truth map and pose; same caveat as nav_eval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

START = (-3.75, 0.73)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="grscenes:MV7J6NIKTKJZ2AABAAAAADQ8_usd")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--min-distance", type=float, default=2.0)
    p.add_argument("--max-distance", type=float, default=8.0)
    p.add_argument("--max-time", type=float, default=60.0)
    p.add_argument("--goal-tolerance", type=float, default=0.5)
    p.add_argument("--models", default="third_party/jeplo/training/logs/rsl_rl/go2_loco/pretrained/exported")
    p.add_argument("--out", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    from common.cerberus_compat.launcher import add_launcher_args

    add_launcher_args(parser)
    args = parser.parse_args(argv)
    from common.cerberus_worlds.registry import get_backend

    get_backend("plain_usd").prepare_scene_env(args.scene)
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    try:
        return run(args)
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)
    finally:
        app.close()


def run(args) -> int:
    from isaaclab.sensors import RayCasterCfg

    from common.cerberus_eval.nav_loop import NavConfig, run_navigation
    from planning.local_planner.core.pure_pursuit import PurePursuitConfig
    from planning.sampling import sample_pairs
    from plugins.jeplo.pattern import RaySphericalSlicePatternCfg
    from policy.locomotion.eval import scene_env as se
    from scripts.explore_tour import load_grid, nearest_free
    from scripts.jeplo_nav import OFFSET_POS, OFFSET_QUAT_WXYZ, JeploIsaacBackend

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = load_grid(args.scene, None)
    start = nearest_free(grid, *START)
    lidar = RayCasterCfg(
        prim_path="/World/envs/env_0/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=OFFSET_POS, rot=OFFSET_QUAT_WXYZ), ray_alignment="base",
        pattern_cfg=RaySphericalSlicePatternCfg(),
        max_distance=2.0, mesh_prim_paths=["/World/scan_proxy"], update_period=0.02, debug_vis=False)
    env, _ = se.build_loco_env(args.scene, [(start[0], start[1], 0.0, grid.floor_z)], task=se.task_for_embodiment("go2"),
                               seed=args.seed, episode_length_s=args.max_time + 60.0, with_cameras=False,
                               extra_sensors={"jeplo_lidar": lidar})
    os.environ["JEPLO_MODELS"] = args.models
    backend = JeploIsaacBackend(env, se)
    pairs = sample_pairs(grid, n=args.pairs, seed=args.seed, min_distance_m=args.min_distance,
                         max_distance_m=args.max_distance).pairs
    config = NavConfig(max_time_s=args.max_time, goal_tolerance_m=args.goal_tolerance)
    records = run_navigation(backend, grid, pairs, seed=args.seed, config=config,
                             pursuit=PurePursuitConfig(dt=config.dt, goal_tolerance_m=config.goal_tolerance_m))
    for record in records:
        (out / f"pair_{record['pair_index']:02d}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    summary = {
        "policy": "jeplo", "scene": args.scene, "seed": args.seed, "pairs": len(records),
        "success": sum(1 for r in records if r.get("success")),
        "falls": sum(1 for r in records if r.get("outcome") == "fall"),
        "success_rate": (sum(1 for r in records if r.get("success")) / len(records)) if records else 0.0,
        "mean_spl": (sum(float(r.get("spl", 0.0)) for r in records) / len(records)) if records else 0.0,
        "records": [{"pair": r["pair_index"], "outcome": r.get("outcome"), "spl": r.get("spl"),
                     "time_s": r.get("time_s")} for r in records],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
