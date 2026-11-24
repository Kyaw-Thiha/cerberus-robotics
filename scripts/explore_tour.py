#!/usr/bin/env python3
"""Exploration tour on the scene 1 occupancy grid: chained A* goals, pure-pursuit tracking, one continuous run.

Kinematic first (``--mock``, no GPU) and then physics (arm Go2 + retrained policy). The record is written once as a
compressed npz (time, pose XYZW, joints, commands, goals, grid, contacts) so the showcase views can be replayed from
it instead of re-simulating.

    python3 scripts/explore_tour.py --scene grscenes:MV7J6NIKTKJZ2AABAAAAADQ8_usd --mock --seed 0 --goals 4
    /workspace/isaaclab/isaaclab.sh -p scripts/explore_tour.py --scene grscenes:<id> \
        --embodiment go2_arx5 --checkpoint policy/locomotion/checkpoints/go2_arx5_locomotion_ga_v1_candidate.pt \
        --start x,y,yaw --seed 0 --out outputs/scratch/tours/physics_seed0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.cerberus_eval.nav_loop import MockBackend
from planning.local_planner.core.pure_pursuit import PurePursuitConfig
from planning.tour import (
    TourConfig,
    pick_tour_goals,
    run_tour,
    tour_npz_payload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="grscenes:MV7J6NIKTKJZ2AABAAAAADQ8_usd")
    parser.add_argument("--grid-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goals", type=int, default=4)
    parser.add_argument("--start", default="", help="x,y[,yaw]; default: the free cell nearest the grid centre")
    parser.add_argument("--out", default=None)
    parser.add_argument("--mock", action="store_true", help="unicycle backend, no Isaac/GPU")
    parser.add_argument("--embodiment", default="go2_arx5")
    parser.add_argument("--checkpoint", default="policy/locomotion/checkpoints/go2_arx5_locomotion_ga_v1_candidate.pt")
    parser.add_argument("--task", default=None)
    parser.add_argument("--goal-tolerance", type=float, default=0.3)
    parser.add_argument("--min-net", type=float, default=6.0)
    parser.add_argument("--max-total-time", type=float, default=180.0)
    parser.add_argument("--video", action="store_true", help="physics only: capture a CHASE clip during the run")
    return parser


def default_out_dir() -> Path:
    return _REPO_ROOT / "outputs" / "scratch" / "tours" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def load_grid(scene: str, grid_dir: str | None):
    from scripts.nav_eval import scene_grid

    return scene_grid(scene, grid_dir)[0]


def nearest_free(grid, x: float, y: float) -> tuple[float, float]:
    import numpy as np

    from planning.global_planner.core.astar import traversable_mask

    free = np.argwhere(traversable_mask(grid))
    if len(free) == 0:
        raise SystemExit("grid has no free cells")
    centre = np.array([(y - grid.origin_xy[1]) / grid.resolution_m,
                       (x - grid.origin_xy[0]) / grid.resolution_m])
    cell = free[np.argmin(((free - centre) ** 2).sum(axis=1))]
    return grid.centre_of(int(cell[0]), int(cell[1]))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out) if args.out else default_out_dir()
    out.mkdir(parents=True, exist_ok=True)

    if args.mock:
        try:
            grid = load_grid(args.scene, args.grid_dir)
        except (SystemExit, FileNotFoundError, RuntimeError) as exc:
            print(f"[tour] {args.scene}: no grid ({exc}); using the synthetic flat grid", file=sys.stderr)
            from planning.occupancy import OccupancyConfig, build_grid
            from scripts.plan_preview import synthetic_flat

            grid = build_grid(*synthetic_flat(), config=OccupancyConfig.from_yaml(), scene_id="synthetic_flat",
                              floor_z=0.0)
        if args.start:
            parts = [float(v) for v in args.start.split(",")]
            start = nearest_free(grid, parts[0], parts[1])
            yaw = parts[2] if len(parts) > 2 else 0.0
        else:
            ny, nx = grid.grid.shape
            start = nearest_free(grid, *grid.centre_of(ny // 2, nx // 2))
            yaw = 0.0
        backend = MockBackend(1)
        backend.reset([(start[0], start[1], yaw)])
        extra = None
    else:
        from policy.locomotion.eval import scene_env as se
        from scripts.nav_eval import IsaacBackend

        grid = load_grid(args.scene, args.grid_dir)
        if not args.start:
            raise SystemExit("physics mode needs --start x,y,yaw (the admitted spawn)")
        parts = [float(v) for v in args.start.split(",")]
        start = nearest_free(grid, parts[0], parts[1])
        yaw = parts[2] if len(parts) > 2 else 0.0
        task = args.task or se.task_for_embodiment(args.embodiment)
        env, _scene = se.build_loco_env(args.scene, [(start[0], start[1], yaw, grid.floor_z)], task=task,
                                        seed=args.seed, episode_length_s=max(120.0, args.max_total_time + 60.0),
                                        with_cameras=args.video, timings_path=out / "build_timings.json")
        hook = se.VelocityCommandHook(env)
        policy = se.load_policy(env, task=task, checkpoint=args.checkpoint)
        backend = IsaacBackend(env, hook, policy)
        robot = env.unwrapped.scene["robot"]

        def extra():
            data = robot.data
            return {
                "root_pos": [float(v) for v in data.root_pos_w[0].detach().cpu()],
                "root_quat_xyzw": [float(v) for v in data.root_quat_w[0].detach().cpu()],
                "joint_pos": [float(v) for v in data.joint_pos[0].detach().cpu()],
            }

    goals = pick_tour_goals(grid, start, seed=args.seed, n=args.goals, min_total_m=args.min_net)
    if not goals:
        print(f"[tour] no reachable goals from {start}", file=sys.stderr)
        return 1
    print(f"[tour] scene={args.scene} start={tuple(round(v, 3) for v in (start[0], start[1], yaw))} "
          f"goals={[(round(x, 2), round(y, 2)) for x, y in goals]}", flush=True)

    config = TourConfig(goal_tolerance_m=args.goal_tolerance, max_total_time_s=args.max_total_time)
    result = run_tour(backend, grid, goals, config=config,
                      pursuit=PurePursuitConfig(dt=config.dt, goal_tolerance_m=config.goal_tolerance_m),
                      extra=extra)
    payload = tour_npz_payload(grid, result, scene_id=args.scene, extra_arrays=(
        {"joint_pos": np.asarray([row.get("joint_pos") for row in result.rows], dtype=np.float32)}
        if result.rows and "joint_pos" in result.rows[0] else None))
    np.savez_compressed(out / "tour.npz", **payload)
    summary = {
        "scene": args.scene, "seed": args.seed, "mode": "mock" if args.mock else "isaac",
        "embodiment": None if args.mock else args.embodiment,
        "start": [start[0], start[1], yaw], "goals": goals,
        "reached": result.reached, "final_distances_m": result.final_distances_m,
        "net_displacement_m": result.net_displacement_m, "path_length_m": result.path_length_m,
        "duration_s": result.duration_s, "fallen": result.fallen, "notes": result.notes,
        "steps": len(result.rows), "npz": str(out / "tour.npz"),
    }
    (out / "tour.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    ok = result.success and result.net_displacement_m >= args.min_net
    print(("PASS " if ok else "FAIL ") + f"tour net={result.net_displacement_m:.2f} m")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
