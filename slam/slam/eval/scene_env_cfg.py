"""Runtime edits to a SLAM-eval env cfg: a GRScenes scene instead of the stock ground, and a mission-shaped episode.

Applied inside the loop's hydra ``main(env_cfg, agent_cfg)`` (sim_ground_truth_loop.py), not through one gym id
per scene. ``isaaclab`` is only imported inside :func:`apply_scene`, so the rest imports (and is tested) without it.

Isaac Lab 3.0 facts these edits rely on, read at release/3.0.0 @ dcabdcda in SOURCE only
(# UNVERIFIED: none of it has run; the pod smoke test must confirm them):

* ``TerrainImporterCfg(terrain_type="usd")`` spawns the USD at ``<prim_path>/terrain`` with no material changes and
  puts the env origins on a grid (``InteractiveScene`` fills ``num_envs``/``env_spacing``), so one env sits at the
  world origin and the scene USD frame equals the world frame.
* The Go2 flat cfg already nulls ``scene.height_scanner``, ``observations.policy.height_scan`` and
  ``curriculum.terrain_levels`` and sets ``terrain_generator = None``: nothing else references the generator
  terrain. :func:`apply_scene` re-checks that rather than assuming it.
* The raw GRScenes house cannot be ray-cast: over a third of its Mesh prims are empty and
  ``MultiMeshRayCaster._load_target_prim_warp_mesh`` raises on the first one (seen on a pod). The LiDAR therefore targets
  ``LIDAR_PROXY_PRIM_PATH``, a static triangle mesh with only the valid world-frame geometry (``scripts/build_lidar_proxy.py``),
  spawned at the world origin like the scene. ``UsdFileCfg(visible=False)`` makes the spawner call
  ``UsdGeom.Imageable.MakeInvisible()`` (sim/utils/prims.py ``clone``), so cameras do not render it, while the ray caster reads
  the Mesh attributes and never checks visibility (base_multi_mesh_ray_caster.py, utils/mesh.py).
* ``MultiMeshRayCaster`` shares ``RayCasterData`` with the plain ``RayCaster``, so ``data.ray_hits_w`` is a
  ``ProxyArray`` resolving to ``(N, B, 3)`` (misses are ``inf``) exactly as before.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from common.cerberus_compat.quat import to_isaac
from common.cerberus_worlds.admission import SpawnPose
from common.cerberus_worlds.scene_geometry import lidar_proxy_path

GROUND_PRIM_PATH = "/World/ground"
"""Where the stock flat cfg puts its terrain; the scene USD takes the same slot so nothing else re-points."""

LIDAR_PROXY_PRIM_PATH = "/World/lidar_proxy"
"""Where the invisible LiDAR proxy mesh is spawned; the only ray-cast target of a scene run."""

EPISODE_MARGIN_S = 10.0
"""Slack added to the mission length so a timeout reset can never teleport the robot mid-run."""

NEVER_S = 1.0e9
"""Command resampling interval that never elapses."""

# Same dome light the plain_usd backend gives a scene (the stock HDR sky is a Nucleus texture).
SCENE_LIGHT_INTENSITY = 2000.0
SCENE_LIGHT_COLOR = (0.75, 0.75, 0.75)


def apply_scene(env_cfg: Any, scene: dict[str, Any], spawn: SpawnPose) -> None:
    """Replace the stock ground with the scene USD and start the robot at the validated ``spawn``.

    ``scene`` is the dict ``plain_usd``'s ``load_scene`` returns (its own ``spawn_pose`` entry is an unvalidated
    placeholder and is ignored). The LiDAR, if the rig has one, becomes a ``MultiMeshRayCasterCfg`` over the scene's
    LiDAR proxy mesh (``lidar_proxy.usd`` beside the scene USD, built by ``scripts/build_lidar_proxy.py``; raises
    FileNotFoundError if it is missing): the raw house has empty meshes the ray caster cannot parse.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sensors import MultiMeshRayCasterCfg
    from isaaclab.terrains import TerrainImporterCfg

    usd_path = scene.get("usd_path")
    if not usd_path:
        raise ValueError(f"scene {scene.get('scene_id')!r} has no usd_path (the stock scene needs no apply_scene)")
    scene_cfg = env_cfg.scene
    _require_no_generator_terrain_users(env_cfg)
    lidar = getattr(scene_cfg, "lidar", None)
    proxy = lidar_proxy_path(usd_path)
    if lidar is not None and not proxy.is_file():
        raise FileNotFoundError(f"LiDAR proxy mesh {proxy} is missing: run scripts/build_lidar_proxy.py --scene {scene['scene_id']}")

    scene_cfg.terrain = TerrainImporterCfg(
        prim_path=GROUND_PRIM_PATH,
        terrain_type="usd",
        usd_path=usd_path,
        collision_group=-1,
        debug_vis=False,
    )
    scene_cfg.sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=SCENE_LIGHT_INTENSITY, color=SCENE_LIGHT_COLOR),
    )

    # UNVERIFIED: the spawn is in the scene USD frame, which equals the world frame because the scene is spawned
    # at identity and env 0 sits at the world origin (grid_transforms(1, ...)); sim_ground_truth_loop checks it.
    robot = scene_cfg.robot
    standing_height = robot.init_state.pos[2]
    half_yaw = spawn.yaw / 2.0
    scene_cfg.robot = robot.replace(
        init_state=robot.init_state.replace(
            pos=(spawn.x, spawn.y, spawn.floor_z + standing_height),
            rot=to_isaac((math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))),  # Cerberus w-first -> Lab native
        )
    )

    if lidar is not None:
        scene_cfg.lidar_proxy = AssetBaseCfg(
            prim_path=LIDAR_PROXY_PRIM_PATH,
            spawn=sim_utils.UsdFileCfg(usd_path=str(proxy), visible=False),
        )
        # Every RayCasterCfg field carries over (rate, pattern, offset, alignment); only the target list is new.
        shared = {
            f.name: getattr(lidar, f.name)
            for f in dataclasses.fields(lidar)
            if f.name not in ("class_type", "mesh_prim_paths")
        }
        scene_cfg.lidar = MultiMeshRayCasterCfg(
            **shared,
            mesh_prim_paths=[
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr=LIDAR_PROXY_PRIM_PATH, merge_prim_meshes=True, track_mesh_transforms=False
                )
            ],
        )


def _require_no_generator_terrain_users(env_cfg: Any) -> None:
    """Fail loudly if the base cfg still has terms that need the removed generator terrain."""
    if getattr(env_cfg.scene, "height_scanner", None) is not None:
        raise ValueError("scene.height_scanner raycasts the generator terrain; the SLAM eval cfg must not have it")
    if getattr(getattr(env_cfg, "curriculum", None), "terrain_levels", None) is not None:
        raise ValueError("curriculum.terrain_levels needs a generator terrain; the SLAM eval cfg must not have it")


def apply_mission(env_cfg: Any, duration_s: float) -> None:
    """One env, a deterministic start, and a velocity command that only the scripted schedule controls.

    The stock velocity task resets the robot to a random pose/joint scale, times the episode out after 20 s (the
    household mission is 21 s with its settle time), resamples the command every 10 s, and replaces the commanded
    yaw rate with a heading controller on every env. Any of those would silently corrupt the canned schedule or
    teleport the robot mid-mission.
    """
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = duration_s + EPISODE_MARGIN_S

    command = env_cfg.commands.base_velocity
    command.heading_command = False
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (NEVER_S, NEVER_S)
    command.debug_vis = False

    reset_base = env_cfg.events.reset_base
    reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    reset_base.params["velocity_range"] = {
        axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)


def physics_summary(env_cfg: Any) -> dict[str, Any]:
    """The physics backend and actuator mode the env cfg will run with; raises unless it is PhysX + legacy actuators.

    Every Cerberus checkpoint was trained on PhysX with the pre-3.0 actuator path, and the stock Go2 task defaults to
    Newton (see common/cerberus_compat/physics.py). A missing ``sim.physics`` is the Lab 2.x single-backend case.
    """
    physics = getattr(env_cfg.sim, "physics", None)
    name = "PhysxCfg (Lab 2.x, only backend)" if physics is None else type(physics).__name__
    use_newton_actuators = bool(getattr(env_cfg.sim, "use_newton_actuators", False))
    if "physx" not in name.lower():
        raise RuntimeError(f"SLAM eval needs PhysX physics but the env cfg resolved to {name}")
    if use_newton_actuators:
        raise RuntimeError("SLAM eval needs the legacy actuator path but sim.use_newton_actuators is True")
    return {"physics_cfg": name, "use_newton_actuators": use_newton_actuators}
