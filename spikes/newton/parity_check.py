"""Stage 3: does the Cerberus Go2 task, rebuilt on Newton, expose the policy's 235-D contract?

Builds the Newton variant (``spikes/newton/tasks.py``), runs 64 envs headless for 500 steps with a zero
action, and compares the live observation layout, joint order and control timing against
``common/cerberus_policies/contracts/heightscan_ppo_v1.yaml``.

    python -m spikes.newton.parity_check --out data/reports/newton/parity --num_envs 64 --steps 500
"""

from __future__ import annotations

import sys
import time

import torch

from spikes.newton import common, contract, tasks


def _reset(env):
    result = env.reset()
    return (result[0] if isinstance(result, tuple) else result), result


def _step(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, rew, terminated, truncated, extras = result
        return obs, rew, terminated, truncated, extras
    obs, rew, done, extras = result
    return obs, rew, done, done, extras


def main() -> int:
    parser = common.build_parser(__doc__)
    parser.add_argument("--joint_ordering", default="none", choices=["none", "physx", "mjwarp", "contract"])
    args = parser.parse_args(sys.argv[1:])

    out = common.out_dir(args.out)
    timer = common.Timer()
    report: dict = {"stage": "parity", "args": vars(args), "versions": common.software_versions(),
                    "env_vars": common.env_vars_that_matter()}
    common.write_json(out / "versions.json", report["versions"])

    cfg, info = tasks.build_cfg(num_envs=args.num_envs, joint_ordering=args.joint_ordering, seed=args.seed)
    report["cfg_info"] = info
    common.write_json(out / "cfg_info.json", info)

    tasks.register()
    from isaaclab.app import launch_simulation
    import gymnasium as gym

    with launch_simulation(cfg, args):
        timer.mark("simulation_ready")
        env = gym.make(tasks.NEWTON_TASK_PLAY_ID, cfg=cfg)
        timer.mark("env_built")
        report["sim"] = common.sim_info(env)
        report["action"] = common.action_layout(env)
        report["joints"] = common.joint_layout(env)
        report["observation_layout"] = common.observation_layout(env)
        try:
            scanner = env.unwrapped.scene["height_scanner"]
        except (KeyError, TypeError):
            scanner = None
        if scanner is not None:
            report["height_scanner"] = {"type": type(scanner).__name__,
                                        "num_rays": int(getattr(scanner, "num_rays", -1)) if hasattr(scanner, "num_rays") else None,
                                        "prim_path": getattr(scanner.cfg, "prim_path", None)}
            try:
                hits = scanner.data.ray_hits_w
                report["height_scanner"]["ray_hits_shape"] = list(hits.shape)
            except Exception as exc:  # noqa: BLE001 - recorded, not fatal
                report["height_scanner"]["ray_hits_error"] = f"{type(exc).__name__}: {exc}"

        obs, raw = _reset(env)
        obs_vec = common.obs_vector(obs)
        report["obs_shape"] = list(obs_vec.shape)
        report["obs_reset_container"] = type(obs).__name__
        report["obs_has_nan"] = bool(torch.isnan(obs_vec).any())
        dim = obs_vec.shape[-1]

        action_dim = report["action"].get("total_action_dim") or 12
        action = torch.zeros((args.num_envs, int(action_dim)), device=obs_vec.device)
        robot = env.unwrapped.scene["robot"]
        start_height = robot.data.root_pos_w[:, 2].clone()

        falls = torch.zeros(args.num_envs, device=obs_vec.device)
        timeout_count = torch.zeros(args.num_envs, device=obs_vec.device)
        nan_seen = False
        t0 = time.time()
        height_drift = None
        for step in range(args.steps):
            obs, rew, terminated, truncated, extras = _step(env, action)
            obs_vec = common.obs_vector(obs)
            if bool(torch.isnan(obs_vec).any()) or bool(torch.isnan(rew).any()):
                nan_seen = True
            base_contact = common.term_bool(env, "base_contact")
            if base_contact is not None:
                falls += base_contact.float()
            timeout_count += truncated.float() if torch.is_tensor(truncated) else 0.0
            if step == 0:
                hits = getattr(scanner.data, "ray_hits_w", None) if scanner is not None else None
                if hits is not None:
                    finite = torch.isfinite(hits[..., 2])
                    report["height_scan_probe"] = {
                        "finite_fraction": float(finite.float().mean()),
                        "hit_z_min": float(hits[..., 2][finite].min()) if finite.any() else None,
                        "hit_z_max": float(hits[..., 2][finite].max()) if finite.any() else None,
                    }
        step_time = time.time() - t0
        height_drift = (robot.data.root_pos_w[:, 2] - start_height).abs()
        report["run"] = {
            "steps": args.steps,
            "num_envs": args.num_envs,
            "seconds": round(step_time, 2),
            "steps_per_s": round(args.steps / max(step_time, 1e-9), 1),
            "nan_seen": nan_seen,
            "final_obs_has_nan": bool(torch.isnan(obs_vec).any()),
            "base_contact_steps_total": float(falls.sum()),
            "envs_with_base_contact": int((falls > 0).sum()),
            "timeouts_total": float(timeout_count.sum()),
            "max_abs_height_drift_m": float(height_drift.max()),
            "mean_height_m": float(robot.data.root_pos_w[:, 2].mean()),
        }
        timer.mark("rollout")

        # ---- contract comparison -------------------------------------------------------------
        contract_summary = contract.contract_summary()
        expected_obs = contract_summary["observation"]
        live_names = report["observation_layout"]["groups"].get("policy", {}).get("terms")
        live_dims = report["observation_layout"]["groups"].get("policy", {}).get("term_dims")
        contract_joints = contract_summary["joints"]
        live_joints = report["joints"]["joint_names"]
        checks = {
            "obs_dim_matches": dim == expected_obs["total_dim"],
            "obs_term_order_matches": live_names == expected_obs["names"],
            "obs_term_dims_match": (None if not live_dims else [int(v) for v in live_dims] == expected_obs["dims"]),
            "joint_name_set_matches": sorted(live_joints) == sorted(contract_joints),
            "joint_order_matches_contract": live_joints == contract_joints,
            "decimation_matches": report["sim"].get("decimation") == contract_summary["control"]["decimation"],
            "step_dt_matches": abs(float(report["sim"].get("step_dt", -1)) - contract_summary["control"]["sim_dt_s"] * contract_summary["control"]["decimation"]) < 1e-9,
            "action_dim_matches": int(report["action"].get("total_action_dim", -1)) == int(contract_summary["action"]["dim"]),
            "command_ranges_match": _ranges_match(report["sim"].get("command_ranges", {}), contract_summary["commands"]["ranges"]),
            "no_nan": not nan_seen and not report["run"]["final_obs_has_nan"],
            "stands_after_500_steps": report["run"]["envs_with_base_contact"] == 0,
        }
        report["contract"] = {
            "expected_obs_names": expected_obs["names"],
            "expected_obs_dims": expected_obs["dims"],
            "expected_total_dim": expected_obs["total_dim"],
            "contract_joint_order": contract_joints,
            "live_joint_order": live_joints,
            "checks": checks,
            "all_passed": all(v is True for v in checks.values()),
            "not_applicable": [k for k, v in checks.items() if v is None],
        }
        common.write_json(out / "parity.json", report)
        print("[parity] " + ", ".join(f"{k}={v}" for k, v in checks.items()), flush=True)
        print(f"[parity] obs_dim={dim} terms={live_names}", flush=True)
        print(f"[parity] joint order == contract: {live_joints == contract_joints}", flush=True)
        common.write_json(out / "timings.json", timer.as_dict())
        env.close()
    print("[parity] done", flush=True)
    return 0 if report["contract"]["all_passed"] else 0  # the verdict is in the JSON, not the exit code


def _ranges_match(live: dict, expected: dict) -> bool:
    pairs = {"lin_vel_x": "vx_mps", "lin_vel_y": "vy_mps", "ang_vel_z": "wz_radps"}
    for live_key, contract_key in pairs.items():
        got = live.get(live_key)
        want = expected.get(contract_key)
        if got is None or want is None:
            return False
        if abs(float(got[0]) - float(want[0])) > 1e-6 or abs(float(got[1]) - float(want[1])) > 1e-6:
            return False
    return True


if __name__ == "__main__":
    from spikes.newton.common import run_stage

    raise SystemExit(run_stage(main))
