#!/usr/bin/env python3
"""Kinematic navigation upper bound: A* + pure pursuit on a unicycle, no physics.

Runs on an occupancy grid (``--grid occupancy.npz``) or, without scene assets, on the deterministic synthetic flat
used by ``scripts/plan_preview.py``. Reports success rate and SPL -- the ceiling the physics runs are compared to.

Usage:
    python3 scripts/kinematic_eval.py [--grid data/scenes/.../occupancy.npz] [--pairs 20] [--seed 0]
                                      [--out outputs/kinematic]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from common.cerberus_eval.nav_metrics import (  # noqa: E402
    aggregate,
    write_aggregate_csv,
    write_run_json,
)
from planning.kinematic_sim import KinematicConfig, run_episode  # noqa: E402
from planning.local_planner.core.pure_pursuit import PurePursuitConfig  # noqa: E402
from planning.occupancy import OccupancyConfig, OccupancyGrid, build_grid, load_npz  # noqa: E402
from planning.sampling import sample_pairs  # noqa: E402
from planning.viz import save_plan_png  # noqa: E402


def synthetic_grid() -> OccupancyGrid:
    from scripts.plan_preview import synthetic_flat

    return build_grid(*synthetic_flat(), config=OccupancyConfig.from_yaml(), scene_id="synthetic_flat", floor_z=0.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", default=None, help="occupancy .npz (default: the synthetic flat)")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--max-distance", type=float, default=10.0)
    parser.add_argument("--max-time", type=float, default=120.0)
    parser.add_argument("--out", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out) if args.out else _REPO_ROOT / "outputs" / "kinematic" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)

    grid = load_npz(args.grid) if args.grid else synthetic_grid()
    stats = sample_pairs(grid, n=args.pairs, seed=args.seed,
                         min_distance_m=args.min_distance, max_distance_m=args.max_distance)
    config = KinematicConfig(max_time_s=args.max_time)
    pursuit = PurePursuitConfig(dt=config.dt, goal_tolerance_m=config.goal_tolerance_m)

    records = []
    paths = []
    for i, pair in enumerate(stats.pairs):
        record = run_episode(grid, pair, config=config, pursuit=pursuit)
        record.update({"seed": args.seed, "pair_index": i})
        records.append(record)
        if record["outcome"] in ("success", "collision"):
            from planning.global_planner.core.astar import astar

            plan = astar(grid, pair.start, pair.goal)
            if plan.success:
                paths.append(plan.path)

    for record in records:
        write_run_json(out / f"{record['scene']}_seed{args.seed}_pair{record['pair_index']}.json", record)
    write_aggregate_csv(out / "kinematic_aggregate.csv", records)
    save_plan_png(grid, out / "kinematic_paths.png", paths=paths,
                  title=f"kinematic: {aggregate(records)['success_rate']:.0%} success, {len(records)} pairs")

    summary = {
        "grid": args.grid or "synthetic_flat",
        "scene": grid.scene_id,
        "seed": args.seed,
        "sampling": {
            "requested": stats.requested, "attempts": stats.attempts,
            "rejected_unreachable": stats.rejected_unreachable, "rejected_distance": stats.rejected_distance,
            "rejected_too_close": stats.rejected_too_close, "pairs": len(stats.pairs),
        },
        "aggregate": aggregate(records),
        "config": {"kinematic": config.__dict__, "pursuit": pursuit.__dict__},
        "out": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
