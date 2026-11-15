"""Stage 5: load one GRScenes home in kit-less Newton, with the Go2 standing at a validated spawn pose.

Two separate steps, deliberately separate processes:

``--probe-only``
    Open the scene USD with the kit-less OpenUSD runtime and *parse* it: report the OpenUSD version,
    prim counts and the rigid bodies that carry several colliders -- the shape the Known Issues entry
    ("OpenUSD can crash while parsing collider-rich rigid bodies", fixed in OpenUSD 26.05) is about.
    A crash here is an OpenUSD/importer failure, not a physics one.

default
    Build the Newton task with the home as the terrain (``terrain_type="usd"``), spawn the Go2 at the
    scene's first admitted spawn pose and step 200 times with a zero action.

    python -m spikes.newton.scene_load --probe-only --usd <scene.usd> --out <dir>
    python -m spikes.newton.scene_load --usd <scene.usd> --out <dir> --steps 200 --mdl-path <Materials>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from spikes.newton import common, tasks


def _probe(usd_path: str, out: Path) -> dict:
    from pxr import Usd, UsdPhysics

    t0 = time.time()
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Usd.Stage.Open returned None for {usd_path}")
    prims = list(stage.Traverse())
    rigid = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    bodies: list[dict] = []
    for body in rigid:
        colliders = [c for c in Usd.PrimRange(body) if c.HasAPI(UsdPhysics.CollisionAPI)]
        if colliders:
            bodies.append({"path": str(body.GetPath()), "colliders": len(colliders)})
    bodies.sort(key=lambda row: row["colliders"], reverse=True)
    report = {
        "usd_path": str(usd_path),
        "size_bytes": Path(usd_path).stat().st_size,
        "openusd_version": ".".join(str(v) for v in Usd.GetVersion()),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "prims": len(prims),
        "rigid_bodies": len(rigid),
        "rigid_bodies_with_colliders": len(bodies),
        "max_colliders_per_body": bodies[0]["colliders"] if bodies else 0,
        "top_bodies": bodies[:10],
        "seconds": round(time.time() - t0, 2),
        "parse_ok": True,
    }
    try:
        import importlib.metadata as md

        report["usd_exchange"] = md.version("usd-exchange")
    except Exception:  # noqa: BLE001 - optional
        report["usd_exchange"] = None
    return report


def _make_cfg(usd_path: str, spawn: tuple[float, float, float] | None, num_envs: int, steps: int):
    import isaaclab.sim as sim_utils
    from isaaclab.terrains import TerrainImporterCfg

    cfg, info = tasks.build_cfg(num_envs=num_envs, seed=0, episode_length_s=steps * 0.02 + 2.0)
    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=str(usd_path),
        debug_vis=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    if spawn is not None:
        x, y, z, yaw = spawn
        cfg.scene.robot.init_state.pos = (x, y, z)
        cfg.scene.robot.init_state.rot = _yaw_quat(yaw)
        info["spawn"] = {"pos": [x, y, z], "yaw_rad": yaw}
    else:
        info["spawn"] = None
    return cfg, info


def _yaw_quat(yaw: float):
    import math

    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))  # wxyz for init_state.rot


def main() -> int:
    from spikes.newton.common import build_parser

    parser = build_parser(__doc__)
    parser.add_argument("--usd", required=True)
    parser.add_argument("--mdl-path", default=None)
    parser.add_argument("--spawn_index", type=int, default=0)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    out = common.out_dir(args.out)

    if args.mdl_path:
        import os

        os.environ["MDL_SYSTEM_PATH"] = str(args.mdl_path)

    if args.probe_only:
        report = {"stage": "scene_probe", "versions": common.software_versions()}
        report.update(_probe(args.usd, out))
        common.write_json(out / "usd_probe.json", report)
        print(f"[scene] usd parse ok: {report['prims']} prims, {report['rigid_bodies']} rigid bodies, "
              f"max {report['max_colliders_per_body']} colliders/body", flush=True)
        return 0

    timer = common.Timer()
    report = {"stage": "scene_load", "args": vars(args), "versions": common.software_versions()}

    spawn = None
    scene_id = Path(args.usd).parent.name
    source = scene_id.split(":")[0] if ":" in scene_id else "grscenes"
    try:
        from common.cerberus_worlds.admission import spawn_pose

        pose = spawn_pose(source, scene_id, args.spawn_index)
        spawn = (pose.x, pose.y, pose.floor_z + 0.4, pose.yaw)
        report["spawn_source"] = "admission manifest"
    except Exception as exc:  # noqa: BLE001 - recorded; the load test still runs from the origin
        report["spawn_source"] = f"origin (admission manifest unavailable: {type(exc).__name__}: {exc})"
    report["spawn"] = spawn

    cfg, info = _make_cfg(args.usd, spawn, args.num_envs, args.steps)
    report["cfg_info"] = info
    tasks.register()

    from isaaclab.app import launch_simulation
    import gymnasium as gym
    import torch

    with launch_simulation(cfg, args):
        timer.mark("simulation_ready")
        env = gym.make(tasks.NEWTON_TASK_PLAY_ID, cfg=cfg)
        timer.mark("env_built")
        report["sim"] = common.sim_info(env)
        robot = env.unwrapped.scene["robot"]
        scanner = env.unwrapped.scene.sensors.get("height_scanner") if env.unwrapped.scene.sensors else None
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs_vec = common.obs_vector(obs)
        start = robot.data.root_pos_w.clone()
        action = torch.zeros((args.num_envs, int(common.action_layout(env).get("total_action_dim") or 12)),
                             device=obs_vec.device)
        t0 = time.time()
        falls = 0.0
        nan_seen = False
        for _ in range(args.steps):
            result = env.step(action)
            obs_vec = common.obs_vector(result[0])
            base_contact = common.term_bool(env, "base_contact")
            if base_contact is not None:
                falls += float(base_contact.float().sum())
            nan_seen = nan_seen or bool(torch.isnan(obs_vec).any()) or bool(torch.isnan(result[1]).any())
        wall = time.time() - t0
        timer.mark("rollout")
        end = robot.data.root_pos_w
        report["run"] = {
            "steps": args.steps, "num_envs": args.num_envs, "seconds": round(wall, 2),
            "base_contact_steps": falls, "nan_seen": nan_seen,
            "start_pos": [round(float(v), 3) for v in start[0].tolist()],
            "end_pos": [round(float(v), 3) for v in end[0].tolist()],
            "horizontal_drift_m": round(float((end[0, :2] - start[0, :2]).norm()), 3),
            "drop_m": round(float(start[0, 2] - end[0, 2]), 3),
        }
        if scanner is not None:
            hits = scanner.data.ray_hits_w
            finite = torch.isfinite(hits[..., 2])
            report["height_scanner"] = {"finite_fraction": float(finite.float().mean()),
                                        "type": type(scanner).__name__}
        common.write_json(out / "scene_load.json", report)
        common.write_json(out / "timings.json", timer.as_dict())
        print(f"[scene] loaded and stepped {args.steps}: drift={report['run']['horizontal_drift_m']}m "
              f"drop={report['run']['drop_m']}m falls={falls}", flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    from spikes.newton.common import run_stage

    raise SystemExit(run_stage(main))
