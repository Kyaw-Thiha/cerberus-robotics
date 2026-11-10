#!/usr/bin/env python3
"""E1 navigation eval: A* + pure pursuit through the frozen Go2 policy in GRScenes homes.

Per scene: sample seeded start-goal pairs on the ground-truth occupancy grid, reset each env to its start, plan with
A*, follow with pure pursuit, drive the policy through the velocity-command hook, and record success, SPL, fall,
timeout, planner failure, static contacts and time. Clips (first success + first failure) are replayed after the
metrics run so recording cannot perturb the numbers.

Ground-truth pose and map; the height scan is simulator-provided. This is NOT onboard-sensing navigation.

``--mock`` runs the same loop against a unicycle backend with no Isaac, no policy and no GPU -- that is how the
bookkeeping, metrics and guards are checked locally.

    python3 scripts/nav_eval.py --mock --pairs 20            # local, synthetic flat
    /workspace/isaaclab/isaaclab.sh -p scripts/nav_eval.py --scenes grscenes:<id>,... --pairs 20 --checkpoint <pt>
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
    aggregate_by_scene,
    write_aggregate_csv,
    write_run_json,
)
from common.cerberus_eval.nav_loop import MockBackend, NavConfig, run_navigation  # noqa: E402
from common.cerberus_eval.video_clips import (  # noqa: E402
    ClipRecorder,
    clip_filename,
    metrics_filename,
    plan_clips,
)
from planning.local_planner.core.pure_pursuit import PurePursuitConfig  # noqa: E402
from planning.occupancy import from_usd, load_npz  # noqa: E402
from planning.sampling import sample_pairs  # noqa: E402
from planning.viz import save_plan_png  # noqa: E402

DEFAULT_SCENES = "grscenes:MV7J6NIKTKJZ2AABAAAAADQ8_usd,grscenes:MV7J6NIKTKJZ2AABAAAAADY8_usd,MVUCSQAKTKJ5EAABAAAAAAI8_usd"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", default=DEFAULT_SCENES, help="comma-separated grscenes:<folder> ids")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=4, help="parallel episodes; all envs share one scene prim")
    parser.add_argument("--checkpoint", default="policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate.pt")
    parser.add_argument("--task", default=None, help="Isaac task id (default: the Cerberus rough Play task)")
    parser.add_argument("--max-time", type=float, default=120.0, help="per-episode timeout, s")
    parser.add_argument("--goal-tolerance", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--max-distance", type=float, default=10.0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--grid-dir", default=None, help="directory holding <scene>_occupancy.npz (default: beside the USD)")
    parser.add_argument("--video", dest="video", action="store_true", default=True)
    parser.add_argument("--no-video", dest="video", action="store_false")
    parser.add_argument("--video-count", type=int, default=3)
    parser.add_argument("--video-downscale", type=int, default=2)
    parser.add_argument("--video-steps", type=int, default=600, help="frames per clip at 50 Hz = 12 s")
    parser.add_argument("--mock", action="store_true", help="local unicycle backend; no Isaac, no policy")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def default_out_dir() -> Path:
    return _REPO_ROOT / "outputs" / "nav_eval" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def scene_grid(scene_id: str, grid_dir: str | None):
    """The scene's occupancy grid: cached .npz if present, else built from the USD (needs pxr, no Isaac)."""
    from common.cerberus_worlds.registry import get_backend

    info = get_backend("plain_usd").describe_scene(scene_id)
    usd = info.get("usd_path")
    if not usd:
        raise SystemExit(f"{scene_id}: {info.get('error') or 'no USD'}")
    usd = Path(usd)
    candidates = []
    if grid_dir:
        candidates.append(Path(grid_dir) / f"{usd.parent.name}_occupancy.npz")
    candidates.append(usd.parent / "occupancy.npz")
    for path in candidates:
        if path.is_file():
            return load_npz(path), path
    grid = from_usd(usd, scene_id=scene_id)
    npz = usd.parent / "occupancy.npz"
    from planning.occupancy import save_npz

    save_npz(grid, npz)
    return grid, npz


class IsaacBackend:
    """Adapts the scene env to the navigation loop's backend protocol."""

    def __init__(self, env, hook, policy):
        self.env = env
        self.hook = hook
        self.policy = policy
        self.num_envs = env.unwrapped.num_envs

    def reset(self, poses) -> None:
        from policy.locomotion.eval import scene_env as se

        se.reset_envs_to(self.env, poses)

    def step(self, commands) -> None:
        import torch


        device = self.env.unwrapped.device
        self.hook.apply(torch.tensor([[c["vx"], c["vy"], c["wz"]] for c in commands], device=device))
        obs = self.env.get_observations()
        obs, _, _, _ = self.env.step(self.policy(obs))
        self._last_obs = obs

    def poses(self):
        from policy.locomotion.eval import scene_env as se

        return [tuple(float(v) for v in row) for row in se.read_poses(self.env).detach().cpu()]

    def falls(self):
        from policy.locomotion.eval import scene_env as se

        return [bool(v) for v in se.read_fall_flags(self.env).detach().cpu()]


def replay_clip(backend, grid, pairs, pair_index, out_dir: Path, *, name: str, video_steps: int, downscale: int,
                config: NavConfig, pursuit: PurePursuitConfig) -> Path | None:
    """Re-run one episode with the camera on env 0 and write the clip + its metrics JSON."""
    clip = ClipRecorder(out_dir, enabled=True, downscale=downscale)
    clip.begin(name)
    frames = {"n": 0}

    def on_step(_i, _commands):
        if frames["n"] < video_steps:
            clip.frame(backend.env.unwrapped, 0)
            frames["n"] += 1

    records = run_navigation(backend, grid, [pairs[pair_index]], max_episodes=1, config=config,
                             pursuit=pursuit, on_step=on_step)
    clip.end()
    path = out_dir / name
    if records:
        write_run_json(out_dir / metrics_filename(name), records[0])
    return path if path.is_file() else None


def run(args) -> int:
    out = Path(args.out_dir) if args.out_dir else default_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    config = NavConfig(max_time_s=args.max_time, goal_tolerance_m=args.goal_tolerance)
    pursuit = PurePursuitConfig(dt=config.dt, goal_tolerance_m=config.goal_tolerance_m)
    all_records: list[dict] = []
    per_scene: dict[str, dict] = {}

    env = hook = policy = None
    if not args.mock:
        from common.cerberus_compat.launcher import headless_requested  # noqa: F401
        from policy.locomotion.eval import scene_env as se

        task = args.task or se.LOCO_TASK
        if args.num_envs > 5:
            print(f"[nav_eval] WARNING: {args.num_envs} envs share one scene prim; spawns must be far apart", file=sys.stderr)
        first_grid, _ = scene_grid(scenes[0], args.grid_dir)
        stats0 = sample_pairs(first_grid, n=args.num_envs, seed=args.seed,
                              min_distance_m=args.min_distance, max_distance_m=args.max_distance)
        if not stats0.pairs:
            raise SystemExit(f"{scenes[0]}: no reachable start-goal pairs")
        spawns = [(p.start[0], p.start[1], 0.0, first_grid.floor_z) for p in stats0.pairs]
        env, _scene = se.build_loco_env(scenes[0], spawns, task=task, seed=args.seed,
                                        episode_length_s=max(120.0, args.max_time + 30.0),
                                        timings_path=out / "build_timings.json")
        hook = se.VelocityCommandHook(env)
        policy = se.load_policy(env, task=task, checkpoint=args.checkpoint)
        backend = IsaacBackend(env, hook, policy)
    else:
        backend = MockBackend(num_envs=args.num_envs, dt=config.dt)

    for scene_id in scenes:
        if args.mock and args.grid_dir is None and not scene_id.startswith("grscenes:"):
            from scripts.plan_preview import synthetic_flat
            from planning.occupancy import OccupancyConfig, build_grid

            grid = build_grid(*synthetic_flat(), config=OccupancyConfig.from_yaml(), scene_id="synthetic_flat", floor_z=0.0)
        else:
            try:
                grid, npz = scene_grid(scene_id, args.grid_dir)
            except (SystemExit, FileNotFoundError, RuntimeError) as e:
                print(f"[nav_eval] {scene_id}: no grid ({e}); skipping", file=sys.stderr)
                continue
        stats = sample_pairs(grid, n=args.pairs, seed=args.seed,
                             min_distance_m=args.min_distance, max_distance_m=args.max_distance)
        print(f"[nav_eval] {scene_id}: {len(stats.pairs)} pairs from {stats.attempts} attempts "
              f"(unreachable {stats.rejected_unreachable}, distance {stats.rejected_distance})", flush=True)
        records = run_navigation(backend, grid, stats.pairs, seed=args.seed, config=config, pursuit=pursuit)
        for record in records:
            record.setdefault("scene", scene_id)
        all_records.extend(records)
        per_scene[scene_id] = {
            "grid": str(npz) if "npz" in locals() else "synthetic",
            "sampling": {
                "requested": stats.requested, "attempts": stats.attempts, "pairs": len(stats.pairs),
                "rejected_unreachable": stats.rejected_unreachable,
                "rejected_distance": stats.rejected_distance,
                "rejected_too_close": stats.rejected_too_close,
            },
            "aggregate": aggregate(records),
            "suspect_successes": sum(1 for r in records if r.get("suspect_success")),
        }
        save_plan_png(grid, out / f"{scene_id.split(':', 1)[-1]}_nav_paths.png", title=f"{scene_id}: {len(records)} episodes")

        if args.video and not args.mock and records:
            outcomes = [r["outcome"] for r in records]
            for k in plan_clips(outcomes, max_clips=args.video_count):
                name = clip_filename(scene_id, args.seed, outcomes[k])
                clip_path = replay_clip(backend, grid, stats.pairs, records[k]["pair_index"], out / "clips",
                                        name=name, video_steps=args.video_steps, downscale=args.video_downscale,
                                        config=config, pursuit=pursuit)
                per_scene[scene_id].setdefault("clips", []).append(str(clip_path) if clip_path else f"{name} (missing)")

    write_aggregate_csv(out / "nav_aggregate.csv", all_records)
    (out / "runs.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in all_records), encoding="utf-8")
    summary = {
        "mode": "mock" if args.mock else "isaac",
        "seed": args.seed,
        "pairs_requested": args.pairs,
        "scenes": per_scene,
        "aggregate": aggregate_by_scene(all_records),
        "caveat": "ground-truth pose and map; simulator-provided height scan; NOT onboard-sensing navigation",
        "out": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": summary["aggregate"].get("all", {}), "out": str(out)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    dry = "--dry-run" in argv
    parser = build_parser()
    if not dry and "--mock" not in argv:
        from common.cerberus_compat.launcher import add_launcher_args

        add_launcher_args(parser)
    args = parser.parse_args(argv)
    if dry:
        print(json.dumps({"scenes": args.scenes.split(","), "pairs": args.pairs, "seed": args.seed,
                          "num_envs": args.num_envs, "mock": args.mock,
                          "out": str(Path(args.out_dir) if args.out_dir else default_out_dir())}, indent=2))
        return 0
    if args.mock:
        return run(args)

    from common.cerberus_worlds.registry import get_backend

    for scene_id in [s.strip() for s in args.scenes.split(",") if s.strip()]:
        get_backend("plain_usd").prepare_scene_env(scene_id)  # MDL path before the app launches
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        return run(args)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
