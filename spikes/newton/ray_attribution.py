"""Ray attribution: does the Newton height scan see the ground, or the robot standing on it?

Flat ground, the Go2 held at its default joint targets, and the 187-cell scan dumped for two sensor
configurations:

* ``global_world_only = false`` — rays query the live scene BVH, which on Newton includes dynamic bodies;
* ``global_world_only = true``  — rays query only the global world (the terrain), which is the flag the stock
  velocity cfg sets and the Newton variant must preserve.

On flat ground the installed formula (``pos_w_z - ray_hits_z - offset``, offset 0.5) predicts a single value
for every cell, ``base_z - 0.5``. Any cell that differs is a ray that hit something other than the ground;
its position in the 17x11 grid says where, and its hit height says what.

    python -m spikes.newton.ray_attribution --out data/reports/newton/ray_attribution --global_world_only false
"""

from __future__ import annotations

import sys

import torch

from spikes.newton import common, tasks


def main() -> int:
    parser = common.build_parser(__doc__)
    parser.add_argument("--global_world_only", default="false", choices=["true", "false"])
    parser.add_argument("--settle_steps", type=int, default=100)
    args = parser.parse_args(sys.argv[1:])
    out = common.out_dir(args.out)
    timer = common.Timer()

    cfg, info = tasks.build_cfg(num_envs=args.num_envs, joint_ordering="none", seed=args.seed,
                                episode_length_s=args.settle_steps * 0.02 + 5.0,
                                global_world_only=args.global_world_only == "true")
    import isaaclab.sim as sim_utils
    from isaaclab.terrains import TerrainImporterCfg

    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0),
        debug_vis=False)
    report = {"stage": "ray_attribution", "args": vars(args), "cfg_info": info,
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
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs_vec = common.obs_vector(obs)
        action = torch.zeros((args.num_envs, int(common.action_layout(env).get("total_action_dim") or 12)),
                             device=obs_vec.device)
        for _ in range(args.settle_steps):
            result = env.step(action)
            obs_vec = common.obs_vector(result[0])
        timer.mark("settled")

        hits = scanner.data.ray_hits_w
        hits = hits.torch if hasattr(hits, "torch") else hits
        if hits.dim() == 4:  # (N, 1, B, 3)
            hits = hits[:, 0]
        hit_z = hits[..., 2]
        pos_w = scanner.data.pos_w
        pos_w = pos_w.torch if hasattr(pos_w, "torch") else pos_w
        base_z = pos_w[:, 2]
        offset = _scan_offset(env)
        cells = int(hit_z.shape[-1])
        scan = obs_vec[:, -cells:].clone()
        predicted = (base_z - offset).unsqueeze(1).expand_as(scan)

        diff = (scan - predicted).abs()
        within_1cm = diff < 1e-2
        rows, cols = _grid_shape(cfg)
        per_env = []
        for env_i in range(args.num_envs):
            bad = (~within_1cm[env_i]).nonzero(as_tuple=False).flatten().tolist()
            per_env.append({
                "env": env_i,
                "base_z": round(float(base_z[env_i]), 4),
                "predicted": round(float(predicted[env_i, 0]), 4),
                "scan_min": round(float(scan[env_i].min()), 4),
                "scan_max": round(float(scan[env_i].max()), 4),
                "cells_within_1cm": int(within_1cm[env_i].sum()),
                "fraction_within_1cm": round(float(within_1cm[env_i].float().mean()), 4),
                "outlier_cells": [
                    {"cell": c, "row": c // cols, "col": c % cols,
                     "scan": round(float(scan[env_i, c]), 4),
                     "hit_z": round(float(hit_z[env_i, c]), 4),
                     "delta": round(float(scan[env_i, c] - predicted[env_i, c]), 4)}
                    for c in bad[:40]
                ],
                "outlier_hit_z_mean": (round(float(hit_z[env_i][~within_1cm[env_i]].mean()), 4)
                                       if bad else None),
            })

        grid = _ascii_grid(within_1cm[0], rows, cols)
        common.write_text(out / "grid_env0.txt", grid + "\n")
        report.update({
            "global_world_only": args.global_world_only == "true",
            "offset": offset,
            "cells": cells,
            "grid_shape": {"rows": rows, "cols": cols},
            "hit_z_stats": {"min": round(float(hit_z.min()), 4), "max": round(float(hit_z.max()), 4),
                            "mean": round(float(hit_z.mean()), 4),
                            "finite_fraction": round(float(torch.isfinite(hit_z).float().mean()), 4)},
            "fraction_within_1cm_all_envs": round(float(within_1cm.float().mean()), 4),
            "per_env": per_env,
            "hit_identifiers": _hit_identifier_report(scanner),
            "sim": common.sim_info(env),
        })
        common.write_json(out / "ray_attribution.json", report)
        common.write_json(out / "timings.json", timer.as_dict())
        print(f"[ray] global_world_only={report['global_world_only']} base_z={float(base_z[0]):.4f} "
              f"predicted={float(predicted[0, 0]):.4f} cells_within_1cm={int(within_1cm[0].sum())}/{cells}", flush=True)
        print(f"[ray] hit_z min={report['hit_z_stats']['min']} max={report['hit_z_stats']['max']} "
              f"finite={report['hit_z_stats']['finite_fraction']}", flush=True)
        print(grid, flush=True)
        if per_env and per_env[0]["outlier_cells"]:
            print("[ray] env0 outliers: " + ", ".join(
                f"cell {o['cell']} (r{o['row']},c{o['col']}) scan={o['scan']} hit_z={o['hit_z']}"
                for o in per_env[0]["outlier_cells"][:12]), flush=True)
        env.close()
    return 0


def _scan_offset(env) -> float:
    """The offset the height-scan observation term actually uses (installed default: 0.5)."""
    term = getattr(getattr(getattr(env.unwrapped.cfg, "observations", None), "policy", None), "height_scan", None)
    params = dict(getattr(term, "params", {}) or {})
    defaults = dict(getattr(getattr(term, "func", None), "__kwdefaults__", {}) or {})
    return float({**defaults, **params}.get("offset", 0.5))


def _grid_shape(cfg) -> tuple[int, int]:
    """Rows x cols of the grid pattern, from the pattern's own size/resolution."""
    pattern = cfg.scene.height_scanner.pattern_cfg
    resolution = float(pattern.resolution)
    size = list(pattern.size)
    return round(size[0] / resolution) + 1, round(size[1] / resolution) + 1


def _ascii_grid(ok_row: torch.Tensor, rows: int, cols: int) -> str:
    """One character per cell: '.' matches the flat-ground prediction, 'X' is an outlier."""
    flat = ok_row.tolist()
    lines = ["flat-ground grid (row-major, rows -> x, cols -> y); '.' = within 1 cm, 'X' = outlier"]
    for r in range(rows):
        lines.append("  " + "".join("." if flat[r * cols + c] else "X" for c in range(cols)))
    return "\n".join(lines)


def _hit_identifier_report(scanner) -> dict:
    """What the sensor exposes about *what* a ray hit (the Newton BVH query returns no prim ids)."""
    data = scanner.data
    attrs = [name for name in dir(data) if not name.startswith("_")]
    has_prim = [name for name in attrs if "prim" in name.lower() or "body" in name.lower() or "shape" in name.lower()]
    return {"data_attributes": attrs[:40], "identity_attributes": has_prim,
            "note": "newton.intersect_ray returns distance and normal only, so no per-ray shape/prim id "
                    "is available; attribution comes from the hit height and the cell's position"}


if __name__ == "__main__":
    from spikes.newton.common import run_stage

    raise SystemExit(run_stage(main))
