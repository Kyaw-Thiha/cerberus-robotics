"""Height-scan convention: read the installed `mdp.height_scan`, then check the Newton sensor against it.

Two pieces of evidence, both from the pod:

1. the source of the observation term that the environment actually imports
   (``isaaclab.envs.mdp.observations.height_scan``) plus its live parameters (offset, clip) as the observation
   manager configured them;
2. a flat-ground measurement: the Go2 stands on a plane, the Newton BVH ray caster returns hits, and the
   height-scan slice of the observation is compared cell by cell with ``pos_w_z - hit_z - offset`` (clipped).

    python -m spikes.newton.height_scan_check --out data/reports/newton/height_scan
"""

from __future__ import annotations

import inspect
import sys

import torch

from spikes.newton import common, tasks


def main() -> int:
    parser = common.build_parser(__doc__)  # already carries --steps
    args = parser.parse_args(sys.argv[1:])
    out = common.out_dir(args.out)
    timer = common.Timer()

    import isaaclab.envs.mdp.observations as observations

    source = inspect.getsource(observations.height_scan)
    print("[scan] installed source:\n" + source, flush=True)
    common.write_text(out / "height_scan_source.txt", source)

    cfg, info = tasks.build_cfg(num_envs=args.num_envs, joint_ordering="none", seed=args.seed,
                                episode_length_s=args.steps * 0.02 + 5.0)
    # flat ground: keep the height scanner (the stock flat cfg removes it; this check needs it)
    import isaaclab.sim as sim_utils
    from isaaclab.terrains import TerrainImporterCfg

    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    info["terrain"] = "plane"
    report = {"stage": "height_scan", "args": vars(args), "cfg_info": info,
              "versions": common.software_versions()}
    tasks.register()

    from isaaclab.app import launch_simulation
    import gymnasium as gym

    with launch_simulation(cfg, args):
        timer.mark("simulation_ready")
        env = gym.make(tasks.NEWTON_TASK_PLAY_ID, cfg=cfg)
        timer.mark("env_built")
        robot = env.unwrapped.scene["robot"]
        scanner = env.unwrapped.scene["height_scanner"]
        term = _height_scan_term(env)
        offset = float(term.get("offset", 0.5))
        clip = term.get("clip")

        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs_vec = common.obs_vector(obs)
        action = torch.zeros((args.num_envs, int(common.action_layout(env).get("total_action_dim") or 12)),
                             device=obs_vec.device)
        for _ in range(args.steps):
            result = env.step(action)
            obs_vec = common.obs_vector(result[0])
        timer.mark("steps")

        hits = scanner.data.ray_hits_w
        hits = hits.torch if hasattr(hits, "torch") else hits
        pos_w = scanner.data.pos_w
        pos_w = pos_w.torch if hasattr(pos_w, "torch") else pos_w
        hit_z = hits[..., 2]
        base_z = pos_w[:, 2].unsqueeze(1)
        expected = base_z - hit_z - offset
        if clip is not None:
            expected = expected.clamp(float(clip[0]), float(clip[1]))

        num_cells = int(expected.shape[-1])
        measured = obs_vec[:, -num_cells:]
        diff = (measured - expected).abs()
        ground = float(robot.data.root_pos_w[:, 2].mean())

        report.update({
            "formula": "height = sensor.data.pos_w[:, 2] - sensor.data.ray_hits_w[..., 2] - offset "
                       "(then clamped by the observation term's clip)",
            "offset": offset,
            "clip": list(clip) if clip is not None else None,
            "term": term,
            "cells": num_cells,
            "hits_shape": list(hit_z.shape),
            "flat_ground": {
                "base_height_m": round(ground, 4),
                "hit_z_mean": round(float(hit_z.mean()), 6),
                "expected_mean": round(float(base_z.mean()) - float(hit_z.mean()) - offset, 4),
                "measured_mean": round(float(measured.mean()), 4),
                "expected_min": round(float(expected.min()), 4),
                "expected_max": round(float(expected.max()), 4),
                "measured_min": round(float(measured.min()), 4),
                "measured_max": round(float(measured.max()), 4),
                "max_abs_cell_diff": round(float(diff.max()), 6),
                "mean_abs_cell_diff": round(float(diff.mean()), 6),
                "cells_within_1e-4": int((diff < 1e-4).all(dim=1).sum()),
            },
            "sim": common.sim_info(env),
        })
        common.write_json(out / "height_scan.json", report)
        common.write_json(out / "timings.json", timer.as_dict())
        print(f"[scan] offset={offset} clip={clip} cells={num_cells}", flush=True)
        print(f"[scan] base_z={ground:.4f} hit_z_mean={float(hit_z.mean()):.6f} "
              f"measured_mean={float(measured.mean()):.4f} expected_mean={float(base_z.mean()) - float(hit_z.mean()) - offset:.4f}",
              flush=True)
        print(f"[scan] max_abs_cell_diff={float(diff.max()):.3e} envs_matching={int((diff < 1e-4).all(dim=1).sum())}/{diff.shape[0]}",
              flush=True)
        env.close()
    return 0


def _height_scan_term(env) -> dict:
    """The live `height_scan` observation term's parameters (offset, clip, params) as configured."""
    manager = env.unwrapped.observation_manager
    active = getattr(manager, "active_terms", {}) or {}
    found: dict = {}
    groups = active.items() if isinstance(active, dict) else [("policy", active)]
    for group, terms in groups:
        for term in terms:
            name = getattr(term, "name", None) or (term[0] if isinstance(term, (list, tuple)) else None)
            cfg = term[1] if isinstance(term, (list, tuple)) and len(term) > 1 else term
            if name == "height_scan":
                params = dict(getattr(cfg, "params", {}) or {})
                found = {"group": group, "params": params,
                         "clip": getattr(cfg, "clip", None),
                         "scale": getattr(cfg, "scale", None)}
                # the callable's own defaults fill anything the manager left unset
                func = getattr(cfg, "func", None)
                if func is not None:
                    defaults = {k: v for k, v in getattr(func, "__kwdefaults__", {}).items()}
                    found["func_defaults"] = defaults
                    params = {**defaults, **params}
                found["offset"] = params.get("offset", 0.5)
                found["params"] = params
                return found
    raise RuntimeError(f"the environment has no height_scan observation term; groups={list(active)}")


if __name__ == "__main__":
    from spikes.newton.common import run_stage

    raise SystemExit(run_stage(main))
