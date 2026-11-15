"""Newton variants of the Cerberus Go2 rough task, built without changing the PhysX configs.

The PhysX task (`UnitreeGo2RoughCerberusEnvCfg[_PLAY]`, `policy/locomotion/core/platforms/go2/env_cfg.py`)
is imported read-only and re-configured here:

* physics: the shared `RoughPhysicsCfg` preset container is replaced by its concrete `newton_mjwarp`
  backend (`cfg_ref` `resolve_physics_preset`), and the backend-native actuator path is enabled --
  exactly what Lab 3.0's own Go2 cfg does for Newton;
* height scan: `RayCasterCfg` -> `isaaclab_newton.sensors.NewtonRaycastSensorCfg`, which casts against
  the live Newton BVH and therefore ignores `mesh_prim_paths`;
* everything the policy can see (observation terms, action term, decimation, command ranges, terrain
  generator, reward/termination set) is left exactly as the PhysX config defines it.

Nothing in this module is imported by the PhysX path.
"""

from __future__ import annotations

import gymnasium as gym

NEWTON_TASK_ID = "Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Newton-v0"
NEWTON_PLAY_TASK_ID = "Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Newton-Play-v0"

ENTRY_POINT = "spikes.newton.tasks:UnitreeGo2RoughCerberusNewtonEnvCfg"
PLAY_ENTRY_POINT = "spikes.newton.tasks:UnitreeGo2RoughCerberusNewtonEnvCfg_PLAY"

_DEFAULT_JOINT_ORDERING = "none"


def _select_newton_physics(cfg) -> dict:
    """Replace the physics preset container with the concrete Newton backend."""
    physics = cfg.sim.physics
    newton = getattr(physics, "newton_mjwarp", None)
    if newton is None:
        raise RuntimeError(
            "the task's physics preset has no `newton_mjwarp` alternative; "
            f"got {type(physics).__name__} with {sorted(k for k in vars(physics) if not k.startswith('_'))}"
        )
    preset_container = type(physics).__name__
    cfg.sim.physics = newton  # SimulationContext rejects the unresolved preset container
    used_newton_actuators = None
    if hasattr(cfg.sim, "use_newton_actuators"):
        cfg.sim.use_newton_actuators = True
        used_newton_actuators = True
    return {"preset_container": preset_container, "physics": type(newton).__name__,
            "use_newton_actuators": used_newton_actuators}


def _newton_height_scanner(cfg) -> dict:
    """Swap the height scanner for the Newton BVH ray-cast sensor, keeping its pattern and offset."""
    old = cfg.scene.height_scanner
    if old is None:
        return {"height_scanner": None}
    from isaaclab_newton.sensors import NewtonRaycastSensorCfg

    pattern = old.pattern_cfg
    cfg.scene.height_scanner = NewtonRaycastSensorCfg(
        prim_path=old.prim_path,
        offset=old.offset,
        ray_alignment=old.ray_alignment,
        pattern_cfg=pattern,
        update_period=old.update_period,
        debug_vis=False,
        mesh_prim_paths=[],
    )
    cells = getattr(pattern, "resolution", None)
    return {
        "height_scanner": type(cfg.scene.height_scanner).__name__,
        "ray_alignment": old.ray_alignment,
        "pattern": type(pattern).__name__,
        "pattern_resolution": cells,
        "pattern_size": list(getattr(pattern, "size", []) or []),
    }


def apply_joint_ordering(cfg, joint_ordering: str) -> dict:
    """Set the documented transfer flags (`joint_ordering` / `body_ordering`) on the robot asset cfg.

    ``contract`` uses the explicit 12-name list from
    ``common/cerberus_policies/contracts/heightscan_ppo_v1.yaml`` -- the order the PhysX-trained
    checkpoint expects, per ``docs/h2_report.md``.
    """
    value = (joint_ordering or _DEFAULT_JOINT_ORDERING).strip().lower()
    robot = cfg.scene.robot
    if value == "none":
        robot.joint_ordering = None
        robot.body_ordering = None
        return {"joint_ordering": None, "body_ordering": None}
    if value == "contract":
        from spikes.newton.contract import contract_joint_order

        robot.joint_ordering = contract_joint_order()
        robot.body_ordering = None
        return {"joint_ordering": robot.joint_ordering, "body_ordering": None}
    robot.joint_ordering = value
    robot.body_ordering = value
    return {"joint_ordering": value, "body_ordering": value}


def build_cfg(*, num_envs: int = 64, joint_ordering: str = "none", episode_length_s: float | None = None,
              forward_command: float | None = None, seed: int | None = None) -> tuple[object, dict]:
    """The Cerberus rough Play cfg on Newton, with the transfer flags applied.

    Returns ``(cfg, info)`` where ``info`` records every deviation from the PhysX config.
    """
    from policy.locomotion.core.platforms.go2.env_cfg import UnitreeGo2RoughCerberusEnvCfg_PLAY

    cfg = UnitreeGo2RoughCerberusEnvCfg_PLAY()
    info = {"base_cfg": type(cfg).__name__, "num_envs": int(num_envs)}
    info.update(_select_newton_physics(cfg))
    info.update(_newton_height_scanner(cfg))
    info.update(apply_joint_ordering(cfg, joint_ordering))

    cfg.scene.num_envs = int(num_envs)
    cfg.scene.env_spacing = 2.5
    if episode_length_s is not None:
        cfg.episode_length_s = float(episode_length_s)
    if seed is not None:
        cfg.seed = int(seed)
    if forward_command is not None:
        # one fixed forward command for every env: same term, only the ranges change
        vel = cfg.commands.base_velocity
        vel.ranges.lin_vel_x = (float(forward_command), float(forward_command))
        vel.ranges.lin_vel_y = (0.0, 0.0)
        vel.ranges.ang_vel_z = (0.0, 0.0)
        vel.heading_command = False
        vel.rel_standing_envs = 0.0
        vel.rel_heading_envs = 0.0
        info["forward_command_mps"] = float(forward_command)
    return cfg, info


def _register(task_id: str, entry_point: str) -> None:
    from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg

    if task_id in gym.registry:
        return
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": entry_point, "rsl_rl_cfg_entry_point": UnitreeGo2RoughPPORunnerCfg},
    )


def register() -> None:
    _register(NEWTON_TASK_ID, ENTRY_POINT)
    _register(NEWTON_PLAY_TASK_ID, PLAY_ENTRY_POINT)


def UnitreeGo2RoughCerberusNewtonEnvCfg():  # noqa: N802 - gym entry-point naming
    cfg, _ = build_cfg(num_envs=64)
    return cfg


def UnitreeGo2RoughCerberusNewtonEnvCfg_PLAY():  # noqa: N802 - gym entry-point naming
    cfg, _ = build_cfg(num_envs=64)
    return cfg
