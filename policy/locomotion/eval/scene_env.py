"""The frozen Go2 rough policy inside a GRScenes home: env composition, per-env resets, command hook, state reads.

POD-ONLY in practice: every function that touches Isaac imports it inside the call, but this module lives under
``policy.locomotion`` whose ``__init__`` registers the tasks and therefore needs Isaac Lab. The pure parts it relies
on (spawn separation, metrics, assertions, clip selection) live in ``common/cerberus_eval`` and are tested locally.

Why a new composition instead of the navigation env (the H2 blocker, ``docs/reports/scene_eval_phase0.md``):
``plain_usd``'s ``build_env`` makes a 33-D navigation observation with no velocity command, while the policy needs
its 235-D training observation. The policy is frozen and must not be reshaped, so this builds the task it was
trained in -- ``Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0`` -- and changes only what the scene requires:

* ``scene.terrain`` -> the preprocessed scene USD (``TerrainImporterCfg(terrain_type="usd")``);
* the height scanner's ray-cast target -> the scene's ``scan_proxy.usd`` (one clean triangle mesh, ceilings removed,
  because a downward ray from 20 m above the base would otherwise hit the ceiling);
* the terrain curriculum terms -> ``None`` (they index the generator terrain that no longer exists);
* the command manager -> a never-resampling, non-heading command buffer that only :class:`VelocityCommandHook`
  writes;
* the reset events -> deterministic (no random pose, no joint scale).

Everything the policy depends on stays identical: observation group and term order, the Play cfg's noise-off
settings, ``JointPositionAction`` with scale 0.25, decimation 4 at dt 0.005 (50 Hz), PhysX.

POD-CHECK items (cannot be verified without the app; the first sim step in ``scripts/scene_eval.py`` reports them):
terrain spawn frame == world frame; ``env_spacing`` + a shared USD leaves per-env collision filtering unverified, so
spawns must be far apart (:func:`common.cerberus_eval.scene_assertions.check_spawn_separation`); the teleport in
:func:`reset_envs_to` must be visible in the next observation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from common.cerberus_eval.scene_assertions import check_spawn_separation
from common.cerberus_worlds.scene_geometry import scan_proxy_path

LOCO_TASK = "Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0"
SCAN_PROXY_PRIM_PATH = "/World/scan_proxy"
GROUND_PRIM_PATH = "/World/ground"
DEFAULT_EPISODE_S = 120.0
DEFAULT_ENV_SPACING_M = 2.0
MIN_SPAWN_SEPARATION_M = 0.8
SCENE_LIGHT_INTENSITY = 2000.0
SCENE_LIGHT_COLOR = (0.75, 0.75, 0.75)
NEVER_S = 1.0e9


def resolve_scene(scene_id: str) -> dict[str, Any]:
    """``scene_id`` (``grscenes:<folder>``) -> the plain_usd backend's scene dict, through the registry only."""
    from common.cerberus_worlds.registry import get_backend

    return get_backend("plain_usd").load_scene(scene_id)


def prepare_scene(scene_id: str) -> None:
    """Set the process env the scene needs (MDL_SYSTEM_PATH). Call BEFORE the Isaac app launches."""
    from common.cerberus_worlds.registry import get_backend

    get_backend("plain_usd").prepare_scene_env(scene_id)


def apply_loco_scene(env_cfg: Any, scene: dict[str, Any], *, scan_proxy: str | Path | None = None) -> None:
    """Swap the generated terrain for the scene USD and point the height scanner at the scene's scan proxy."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.terrains import TerrainImporterCfg

    usd_path = scene.get("usd_path")
    if not usd_path:
        raise ValueError(f"scene {scene.get('scene_id')!r} has no usd_path (the stock scene is not supported here)")
    proxy = Path(scan_proxy) if scan_proxy is not None else scan_proxy_path(usd_path)
    if not proxy.is_file():
        raise FileNotFoundError(
            f"height-scan proxy {proxy} is missing: run scripts/build_scan_proxy.py --scene {scene.get('scene_id')}"
        )

    scene_cfg = env_cfg.scene
    scene_cfg.terrain = TerrainImporterCfg(
        prim_path=GROUND_PRIM_PATH,
        terrain_type="usd",
        usd_path=str(usd_path),
        collision_group=-1,
        debug_vis=False,
    )
    scene_cfg.sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=SCENE_LIGHT_INTENSITY, color=SCENE_LIGHT_COLOR),
    )
    scene_cfg.scan_proxy = AssetBaseCfg(
        prim_path=SCAN_PROXY_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(usd_path=str(proxy), visible=False),
    )

    scanner = getattr(scene_cfg, "height_scanner", None)
    if scanner is None:
        raise ValueError("the rough Play cfg has no scene.height_scanner; the policy would get no 187-cell scan")
    scanner.mesh_prim_paths = [SCAN_PROXY_PRIM_PATH]

    # The curriculum terms index the generator terrain that scene_cfg.terrain no longer is.
    curriculum = getattr(env_cfg, "curriculum", None)
    for name in ("terrain_levels", "terrain_level_max", "terrain_level_frac_at_max"):
        if curriculum is not None and getattr(curriculum, name, None) is not None:
            setattr(curriculum, name, None)


def apply_deterministic_commands(env_cfg: Any) -> None:
    """One command source (the hook), one start pose (the driver's teleport), no randomisation.

    The stock velocity task resamples the command every 10 s, replaces the commanded yaw rate with a heading
    controller on every env, randomises the reset pose and scales the reset joint positions. Any of those would
    silently override the navigation command or teleport the robot mid-episode.
    """
    command = env_cfg.commands.base_velocity
    command.heading_command = False
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (NEVER_S, NEVER_S)
    command.debug_vis = False

    reset_base = env_cfg.events.reset_base
    reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    reset_base.params["velocity_range"] = {axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")}
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)


def write_timings(path: str | Path | None, timings: dict[str, float]) -> None:
    """Write (or update) the build-stage timings JSON; no behaviour change, diagnostics only."""
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
    existing.update({k: round(float(v), 3) for k, v in timings.items()})
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_loco_env(
    scene_id: str,
    spawn_poses: Sequence[Sequence[float]],
    *,
    task: str = LOCO_TASK,
    num_envs: int | None = None,
    episode_length_s: float = DEFAULT_EPISODE_S,
    seed: int = 0,
    env_spacing: float = DEFAULT_ENV_SPACING_M,
    scan_proxy: str | Path | None = None,
    timings_path: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """``gym.make`` the Play task in the scene and teleport every env to its ``(x, y, yaw[, floor_z])`` spawn.

    Returns ``(wrapped_env, scene)`` where ``wrapped_env`` is the ``RslRlVecEnvWrapper`` the runner and the policy
    expect (same wrapper ``attack_env.build_attack_env`` uses). Call AFTER AppLauncher has started.
    """
    import gymnasium as gym
    import importlib

    import isaaclab_tasks  # noqa: F401  (registers the tasks)
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    importlib.import_module("policy.locomotion")  # by name: `import policy.locomotion` shadows a local name
    from common.cerberus_compat.cfg_registry import load_env_cfg

    n = len(spawn_poses) if num_envs is None else int(num_envs)
    if n != len(spawn_poses):
        raise ValueError(f"num_envs={n} but {len(spawn_poses)} spawn poses were given")
    problems = check_spawn_separation(spawn_poses, MIN_SPAWN_SEPARATION_M)
    if problems:
        raise ValueError("spawn poses are too close for a shared scene: " + "; ".join(problems))

    t0 = time.perf_counter()
    scene = resolve_scene(scene_id)
    t_scene = time.perf_counter()
    env_cfg = load_env_cfg(task)
    t_cfg = time.perf_counter()
    env_cfg.scene.num_envs = n
    env_cfg.scene.env_spacing = float(env_spacing)
    env_cfg.episode_length_s = float(episode_length_s)
    env_cfg.seed = int(seed)
    apply_loco_scene(env_cfg, scene, scan_proxy=scan_proxy)
    apply_deterministic_commands(env_cfg)
    t_apply = time.perf_counter()

    # gym.make is where the scene USD stage loads, the scan-proxy mesh is built and the articulation is
    # initialised. They are one call in Isaac Lab, so they are timed together and named for what they cover.
    raw_env = gym.make(task, cfg=env_cfg)
    t_make = time.perf_counter()
    env = RslRlVecEnvWrapper(raw_env, clip_actions=None)
    env.reset()  # initialises the managers/buffers; the teleport below is what actually places the robots
    t_reset = time.perf_counter()
    reset_envs_to(env, spawn_poses)
    t_teleport = time.perf_counter()

    timings = {
        "scene_resolve_s": t_scene - t0,
        "cfg_load_s": t_cfg - t_scene,
        "scene_apply_s": t_apply - t_cfg,
        "gym_make_s__stage_load_mesh_build_articulation_init": t_make - t_apply,
        "wrapper_and_reset_s": t_reset - t_make,
        "teleport_s": t_teleport - t_reset,
        "build_total_s": t_teleport - t0,
    }
    write_timings(timings_path, timings)
    print("[scene_env] build timings: " + json.dumps({k: round(v, 2) for k, v in timings.items()}), flush=True)
    return env, scene


def reset_envs_to(env: Any, poses: Sequence[Sequence[float]]) -> Any:
    """Teleport the envs named by ``poses``; returns the fresh observation.

    ``poses`` is one entry per env, each ``(x, y, yaw)`` / ``(x, y, yaw, floor_z)`` in the scene/world frame (metres,
    radians) or ``None`` to leave that env running (the navigation loop starts episodes on individual envs). The
    teleport is a write to the sim state, not a randomisation: it is the only way to give each env its own start
    pose, since ``init_state`` is one value for the whole articulation cfg.
    """
    import torch

    from common.cerberus_compat.quat import to_isaac

    raw = env.unwrapped
    n = raw.num_envs
    if len(poses) != n:
        raise ValueError(f"{len(poses)} poses for {n} envs")
    selected = [(i, p) for i, p in enumerate(poses) if p is not None]
    if not selected:
        return env.get_observations()
    device = raw.device
    ids = torch.tensor([i for i, _ in selected], device=device, dtype=torch.long)
    active = [p for _, p in selected]
    asset = raw.scene["robot"]

    default_root = asset.data.default_root_state[ids].clone()
    x = torch.tensor([float(p[0]) for p in active], device=device, dtype=default_root.dtype)
    y = torch.tensor([float(p[1]) for p in active], device=device, dtype=default_root.dtype)
    yaw = torch.tensor([float(p[2]) for p in active], device=device, dtype=default_root.dtype)
    floor_z = torch.tensor([float(p[3]) if len(p) > 3 else 0.0 for p in active], device=device, dtype=default_root.dtype)
    root = default_root.clone()
    root[:, 0] = x
    root[:, 1] = y
    root[:, 2] = default_root[:, 2] + floor_z
    half = yaw / 2.0
    quats = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=1)
    root[:, 3:7] = torch.tensor([to_isaac(tuple(q)) for q in quats.tolist()], device=device, dtype=root.dtype)

    asset.write_root_pose_to_sim(root[:, :7], env_ids=ids)
    asset.write_root_velocity_to_sim(torch.zeros((len(active), 6), device=device, dtype=root.dtype), env_ids=ids)
    asset.write_joint_state_to_sim(
        asset.data.default_joint_pos[ids].clone(),
        torch.zeros((len(active), asset.num_joints), device=device, dtype=root.dtype),
        env_ids=ids,
    )
    raw.scene.write_data_to_sim()
    if getattr(raw, "episode_length_buf", None) is not None:
        raw.episode_length_buf[ids] = 0
    termination = getattr(raw, "termination_manager", None)
    if termination is not None and hasattr(termination, "reset"):
        termination.reset(ids)
    return env.get_observations()


class VelocityCommandHook:
    """The B1 command channel: writes ``vel_command_b`` directly, the buffer the command term itself samples.

    Same mechanism ``adversarial_robustness.attack_env.ManagerBasedHooks.apply_command`` uses for x/y (confirmed in
    H2), extended to the yaw rate. Re-applied every step because the term would otherwise overwrite it on its own
    (long) timer.
    """

    def __init__(self, env: Any, command_name: str = "base_velocity") -> None:
        self.env = env.unwrapped
        self.term = self.env.command_manager.get_term(command_name)

    def apply(self, commands: Any) -> None:
        """``commands``: tensor (num_envs, 3) of ``(vx, vy, wz)``."""
        self.term.vel_command_b[:, :3] = commands

    def zero(self) -> None:
        self.term.vel_command_b[:, :3] = 0.0


def load_policy(env: Any, task: str = LOCO_TASK, checkpoint: str | None = None, device: str | None = None):
    """Load the frozen actor as ``obs -> actions``: the H2 path (``train_adversary.load_frozen_policy``'s sequence).

    ``check_checkpoint_format`` -> ``make_runner`` -> ``runner.load`` -> ``runner.get_inference_policy``, applied to
    the scene env (an ``RslRlVecEnvWrapper``) instead of a throwaway env. Nothing is trained; parameters are frozen
    and the policy is put in eval mode.
    """
    from common.cerberus_compat.cfg_registry import load_agent_cfg
    from common.cerberus_compat.rsl_rl_compat import check_checkpoint_format, make_runner

    if checkpoint is None:
        raise ValueError("a checkpoint path is required")
    agent_cfg = load_agent_cfg(task)
    fmt = check_checkpoint_format(checkpoint)
    print(f"[scene_env] frozen policy checkpoint format: {fmt}")
    runner = make_runner(env, agent_cfg, log_dir=None, device=device or getattr(agent_cfg, "device", "cuda:0"))
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    return policy


def read_poses(env: Any):
    """``(num_envs, 3)`` tensor of ``(x, y, yaw)`` in the world frame."""
    import torch

    from common.cerberus_compat.quat import from_isaac

    data = env.unwrapped.scene["robot"].data
    pos = data.root_pos_w[:, :3]
    q = from_isaac(data.root_quat_w)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack([pos[:, 0], pos[:, 1], yaw], dim=1)


def read_fall_flags(env: Any):
    """Per-env True when a NON-timeout termination term fired (base contact / illegal contact)."""
    import torch

    raw = env.unwrapped
    manager = getattr(raw, "termination_manager", None)
    flags = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    if manager is None:
        return flags
    for name in getattr(manager, "active_terms", []):
        if name == "time_out":
            continue
        try:
            term = manager.get_term(name)
        except Exception as e:  # noqa: BLE001 - a term that cannot be read is not a fall signal
            print(f"[scene_env] WARNING: termination term {name!r} unreadable ({type(e).__name__}: {e})")
            continue
        if term is not None:
            flags |= term.bool()
    return flags


def stand_metrics(env: Any, base_name: str = "base", foot_names: Sequence[str] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")):
    """``{"base_minus_foot_z": tensor}`` or None when the bodies are not in this articulation."""
    asset = env.unwrapped.scene["robot"]
    base_ids = asset.find_bodies(base_name)[0]
    foot_ids = asset.find_bodies(list(foot_names))[0]
    if not base_ids or len(foot_ids) < 4:
        return None
    z = asset.data.body_pos_w[:, :, 2]
    return {"base_minus_foot_z": z[:, base_ids[0]] - z[:, foot_ids].mean(dim=1)}


def nonfinite_obs_rows(obs: Any) -> list[int]:
    """Indices of envs whose observation holds a NaN/inf (torch tensor)."""
    import torch

    if not isinstance(obs, torch.Tensor):
        return []
    return torch.nonzero(~torch.isfinite(obs).all(dim=1)).flatten().tolist()
