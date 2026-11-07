#!/usr/bin/env python3
"""P0 stack check: what is a standing Go2's projected gravity, and which quaternion order does
``robot.data.root_quat_w`` use on THIS Isaac Lab line?

Why: the 2.x -> 3.x move changed Cerberus's quaternion convention (Lab 2.x wxyz, Lab 3.0 xyzw;
see ``common/cerberus_compat/quat.py``). Anything that reorders quaternions by hand -- the ROS
inference node planned in H3, custom reward/observation code -- silently breaks if it assumes the
wrong order. ``projected_gravity`` is the cheapest end-to-end probe: for an upright robot it must
read approximately ``[0, 0, -1]`` in the body frame, and it is derived from the root quaternion, so
it catches a convention error anywhere along that path.

Second job: record the physics backend the env ACTUALLY resolved, and refuse to call the run healthy
if it is not PhysX (Lab 3.0's stock Go2 task defaults to Newton -- see
``common/cerberus_compat/physics.pin_physx``).

Pod-only (needs Isaac Sim). Run it on the pod, e.g.:

    export TERM=xterm ISAACLAB_PATH=/workspace/isaaclab
    $ISAACLAB_PATH/isaaclab.sh -p scripts/check_isaac_quat_convention.py \
        --task Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0 \
        --merge-into outputs/env_report.json --output data/reports/quat_convention.json

Exit codes: 0 pass, 1 fail, 2 bad args.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# How far projected gravity may sit from [0, 0, -1] before we call the stack broken. The robot is
# held at its default pose for SETTLE_S first, so this is generous enough for PD sag but far below
# any axis mix-up (a wrong quaternion order puts a large component on x or y).
PROJECTED_GRAVITY_ATOL = 0.10
SETTLE_S = 1.0


def classify_quat_order(root_quat_row) -> dict:
    """Which index of a standing robot's root quaternion is ~1 -- i.e. the native order.

    ``root_quat_row`` is any 4-element sequence (torch tensor row, list, tuple). A robot spawned
    upright has an identity-ish root rotation ``(1, 0, 0, 0)`` in (w, x, y, z) or ``(0, 0, 0, 1)``
    in (x, y, z, w); whichever slot holds ~1 identifies the convention without trusting the stack's
    own quaternion math. Pure python, no torch: usable in local tests.
    """
    vals = [float(v) for v in root_quat_row]
    if len(vals) != 4:
        raise ValueError(f"expected 4 quaternion components, got {len(vals)}")
    idx = max(range(4), key=lambda i: abs(vals[i]))
    names = ("wxyz", "xyzw")
    convention = "wxyz" if idx == 0 else ("xyzw" if idx == 3 else "ambiguous")
    return {
        "values": vals,
        "argmax_abs": idx,
        "argmax_is_plus_one": vals[idx] > 0,
        "convention": convention,
        "leading_component": vals[idx],
        "ok_in": names,
    }


def evaluate(projected_gravity_rows, quat_rows, atol: float = PROJECTED_GRAVITY_ATOL) -> dict:
    """Judge probe values. Pure: takes per-env lists/rows, returns the verdict dict.

    ``projected_gravity_rows``: sequence of 3-element rows (one per env).
    ``quat_rows``: sequence of 4-element rows (one per env).
    """
    if len(projected_gravity_rows) == 0:
        raise ValueError("no envs to evaluate")
    if len(projected_gravity_rows) != len(quat_rows):
        raise ValueError("projected gravity and quaternion row counts differ")
    worst = 0.0
    worst_env = -1
    for i, pg in enumerate(projected_gravity_rows):
        err = max(abs(float(pg[j]) - (0.0, 0.0, -1.0)[j]) for j in range(3))
        if err > worst:
            worst, worst_env = err, i
    gravity_ok = worst <= atol
    orders = [classify_quat_order(q) for q in quat_rows]
    conventions = {o["convention"] for o in orders}
    # Resolve before building the dict: ``set.pop()`` mutates, and the length check below must see
    # the original cardinality (an earlier version consumed the set inside the dict literal).
    convention = conventions.pop() if len(conventions) == 1 else "inconsistent"
    return {
        "num_envs": len(projected_gravity_rows),
        "projected_gravity_atol": atol,
        "projected_gravity_max_abs_error": worst,
        "projected_gravity_worst_env": worst_env,
        "projected_gravity_ok": gravity_ok,
        "quat_orders": orders,
        "quat_convention": convention,
        "passed": bool(gravity_ok and convention != "inconsistent"),
    }


def physics_summary(env_cfg) -> dict:
    """What the env cfg resolved for physics, plus whether the legacy actuator path is selected."""
    physics = getattr(env_cfg.sim, "physics", None)
    default = getattr(physics, "default", None) if physics is not None else None
    return {
        "physics_default_class": type(default).__name__ if default is not None else None,
        "physics_default_mro": [c.__name__ for c in type(default).__mro__] if default is not None else [],
        "is_physx": any("Physx" in c.__name__ or "PhysX" in c.__name__ for c in type(default).__mro__)
        if default is not None
        else False,
        "use_newton_actuators": getattr(env_cfg.sim, "use_newton_actuators", None),
        "sim_dt": getattr(env_cfg.sim, "dt", None),
        "decimation": getattr(env_cfg, "decimation", None),
    }


def observation_probe(raw_env, group: str = "policy") -> dict:
    """Locate the projected_gravity slice inside the policy observation group and return it.

    Uses the manager's public bookkeeping (``active_terms`` / ``group_obs_term_dim``) rather than
    hardcoding index 6, so a changed term order shows up as a wrong-value failure instead of a
    silently mis-sliced probe.
    """
    manager = raw_env.observation_manager
    names = list(manager.active_terms[group])
    dims = [int(d[0]) for d in manager.group_obs_term_dim[group]]
    if len(names) != len(dims):
        raise RuntimeError(f"active_terms/dim mismatch for group {group!r}: {names} vs {dims}")
    if "projected_gravity" not in names:
        raise RuntimeError(f"group {group!r} has no projected_gravity term: {names}")
    idx = names.index("projected_gravity")
    start = sum(dims[:idx])
    obs = manager.compute()[group]
    return {
        "group": group,
        "term_names": names,
        "term_dims": dims,
        "obs_dim": int(sum(dims)),
        "projected_gravity_slice": [start, start + dims[idx]],
        "rows": obs[:, start : start + dims[idx]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--task",
        default="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0",
        help="Gym task id (the Play variant: no push events, no observation corruption).",
    )
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=_REPO_ROOT / "data" / "reports" / "quat_convention.json")
    parser.add_argument(
        "--merge-into",
        type=Path,
        default=_REPO_ROOT / "outputs" / "env_report.json",
        help="Existing env report to fold the runtime findings into (written by scripts/env_report.py).",
    )
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args(argv)

    import gymnasium as gym
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    result: dict = {"task": args.task, "passed": False}
    exit_code = 1
    try:
        import torch  # noqa: PLC0415

        import isaaclab_tasks  # noqa: F401, PLC0415
        import policy.locomotion  # noqa: F401, PLC0415 -- registers the Cerberus tasks

        from common.cerberus_compat.isaac_data import as_tensor
        from common.cerberus_compat.quat import isaac_quat_convention

        env_cfg = gym.spec(args.task).kwargs["env_cfg_entry_point"]()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = "cuda:0"
        physics = physics_summary(env_cfg)
        result["physics"] = physics
        result["stack_quat_convention"] = isaac_quat_convention()

        env = gym.make(args.task, cfg=env_cfg)
        raw_env = env.unwrapped
        # Hold the default pose: with no policy, the PD controller holds the spawn joint targets, so
        # the base stays upright and both probes read their "standing" values.
        steps = max(1, round(SETTLE_S / raw_env.step_dt))
        obs = env.get_observations() if hasattr(env, "get_observations") else None
        for _ in range(steps):
            obs, _, _, _ = env.step(torch.zeros(raw_env.action_space.shape, device=raw_env.device))

        probe = observation_probe(raw_env, group="policy")
        pg_rows = probe.pop("rows").detach().cpu().tolist()
        quat_rows = as_tensor(raw_env.scene["robot"].data.root_quat_w).detach().cpu().tolist()
        verdict = evaluate(pg_rows, quat_rows)
        result.update(
            {
                "observation": {k: v for k, v in probe.items() if k != "rows"},
                "projected_gravity_first_env": pg_rows[0],
                "root_quat_w_first_env": quat_rows[0],
                **verdict,
                "num_steps_settled": steps,
            }
        )
        # Explicit physics gate: the handoff forbids a silent Newton run.
        result["passed"] = bool(verdict["passed"] and physics.get("is_physx"))
        result["failure_reasons"] = []
        if not verdict["projected_gravity_ok"]:
            result["failure_reasons"].append(
                f"projected_gravity max abs error {verdict['projected_gravity_max_abs_error']:.4f} "
                f"> atol {PROJECTED_GRAVITY_ATOL}"
            )
        if verdict["quat_convention"] == "inconsistent":
            result["failure_reasons"].append("envs disagree on the root quaternion component order")
        if not physics.get("is_physx"):
            result["failure_reasons"].append(
                f"physics backend is {physics.get('physics_default_class')!r}, not PhysX"
            )
        result["failure_reasons"] or result.pop("failure_reasons")
        exit_code = 0 if result["passed"] else 1
        env.close()
    except Exception as e:  # noqa: BLE001 -- report, don't lose the traceback to SimulationApp's exit(0)
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[quat_check] wrote {args.output}")
        print(f"[quat_check] passed={result.get('passed')} physics={result.get('physics')}")
        print(f"[quat_check] projected_gravity(first env)={result.get('projected_gravity_first_env')}")
        print(f"[quat_check] root_quat_w(first env)={result.get('root_quat_w_first_env')}")
        print(f"[quat_check] quat_convention={result.get('quat_convention')}")
        if args.merge_into and args.merge_into.is_file():
            sys.path.insert(0, str(_REPO_ROOT / "scripts"))
            from env_report import merge_runtime_probe  # noqa: PLC0415

            merge_runtime_probe(
                args.merge_into,
                {
                    "resolved_physics": result.get("physics"),
                    "quat_convention": {
                        "root_quat_w": result.get("quat_convention"),
                        "stack": result.get("stack_quat_convention"),
                        "projected_gravity_ok": result.get("projected_gravity_ok"),
                        "passed": result.get("passed"),
                    },
                    "runtime_probe": {"task": result.get("task"), "report": str(args.output)},
                },
            )
            print(f"[quat_check] merged into {args.merge_into}")
        # SimulationApp.close() exits 0 whatever happened (docs/runpod.md); never let it mask a failure.
        sys.stdout.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    sys.exit(main())
