"""Plain-USD world backend: any USD scene + an Isaac Lab articulation, no Arena.

The fallback/primary path for GRScenes homes (W1: "If A fails on multi-room scale only, fall back to
plain_usd_backend ... and keep Arena for kitchens"). Scene discovery is pure python and shared with the Arena
backend (``common/cerberus_worlds/arena_assets/__init__.py``); ``build_env`` builds a minimal navigation
``ManagerBasedRLEnvCfg`` with plain Isaac Lab and needs the Isaac app running. All isaaclab imports are inside
method bodies.

Scene ids: ``grscenes:<folder>`` (folders under ``data/scenes/<source>/``, ``$CERBERUS_SCENES_DIR`` overrides the
root; ``start_result_navigation.usd`` preferred over ``start_result_raw.usd``) and ``stock`` (a ground plane, the
no-data control run).

Env layout: the scene USD is spawned ONCE as a shared prim (``/World/Scene``) and the robot is cloned per env at
``{ENV_REGEX_NS}/Robot`` (``configure(shared_scene=False)`` spawns one scene copy per env instead). With a shared
scene the default ``env_spacing`` is 0: every robot starts at the same spawn point and envs are isolated from each
other by Isaac Lab's per-env collision filtering. # UNVERIFIED: spacing 0 + filter_collisions with a shared USD.

Isaac Lab 2.3.2 (Sim 5.1, Python 3.11) vs the 3.0 line (Sim 6.2, Python 3.12), as handled here:
  * Quaternions in ``init_state.rot`` are WXYZ on 2.3.2 and XYZW on 3.0. ``_is_lab3()`` detects the 3.0 line by
    ``isaaclab_physx`` being importable (``PhysxCfg`` moved there) and ``_quat_from_xyzw`` converts.
  * Articulation/rigid data are torch tensors on 2.3.2 and proxy/warp arrays on 3.0; the waypoint terms read them
    through ``go2_embodiment._to_torch``.
  * ``configclass`` imports from ``isaaclab.utils`` on both; a fallback to ``isaaclab.utils.configclass`` is kept.
  * We never touch ``sim.physics`` (``PhysxCfg`` import path differs), so the version default is used: PhysX.
  * ``--headless`` is deprecated on 3.0 (the app launcher still accepts it).
Verified on a pod: ``build_env`` on the Lab 3.0 GA stack (Sim 6.2) built and rendered a GRScenes home through the
shared-scene path (data/reports/ga_upgrade/render_r3.json, 2026-10-25; also the admission sweeps, 2026-10-11/12).
Not verified: Lab 2.3.2.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import MISSING
from pathlib import Path
from typing import Any

from common.cerberus_description.embodiment import EmbodimentCfg
from common.cerberus_worlds.arena_assets import (
    DEFAULT_SOURCE,
    discover_scenes,
    ensure_mdl_system_path,
    find_scene,
    sim_usd_path,
    spawn_pose_to_background_pose,
)
from common.cerberus_worlds.arena_assets.go2_embodiment import (
    DEFAULT_EPISODE_LENGTH_S,
    DEFAULT_WAYPOINT_RADIUS_M,
    DEFAULT_WAYPOINT_XY,
    SHARED_SCENE_PRIM_PATH,
    make_go2_mdp_cfgs,
    waypoint_reached,
    waypoint_reward,
)
from common.cerberus_worlds.base import SceneInfo, WorldBackend

STOCK_SCENE_ID = "stock"
DEFAULT_ENV_SPACING = 30.0
DEFAULT_ROBOT_Z = 0.4  # UNITREE_GO2_CFG init_state.pos z
SIM_DT = 0.005
DECIMATION = 4  # 50 Hz control, the Cerberus locomotion convention
_ENV_IDS = itertools.count(1)


def waypoint_params(task: Any) -> dict[str, Any]:
    """Resolve the `task` argument of build_env: None -> defaults, dict -> overrides on top of the defaults."""
    params: dict[str, Any] = {
        "waypoint_xy": tuple(DEFAULT_WAYPOINT_XY),
        "radius": DEFAULT_WAYPOINT_RADIUS_M,
        "episode_length_s": DEFAULT_EPISODE_LENGTH_S,
    }
    if task is None:
        return params
    if not isinstance(task, dict):
        raise TypeError(f"plain_usd task must be None or a dict of {sorted(params)}, got {type(task).__name__}")
    unknown = sorted(set(task) - set(params))
    if unknown:
        raise ValueError(f"Unknown plain_usd task keys {unknown}; valid: {sorted(params)}")
    params.update(task)
    params["waypoint_xy"] = tuple(params["waypoint_xy"])
    return params


def _is_lab3() -> bool:
    """True on the Isaac Lab 3.0 line (XYZW quaternions, ``PhysxCfg`` in ``isaaclab_physx``)."""
    try:
        import isaaclab_physx  # noqa: F401
    except ImportError:
        return False
    return True


def _quat_from_xyzw(q_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """XYZW -> the init_state.rot convention of the installed Isaac Lab (WXYZ on 2.3.2)."""
    x, y, z, w = q_xyzw
    return (x, y, z, w) if _is_lab3() else (w, x, y, z)


def _yaw_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class PlainUsdBackend(WorldBackend):
    """Plain Isaac Lab worlds. Extra (non-abstract) helpers: ``configure``, ``normalize_scene_ref``,
    ``describe_scene``, ``prepare_scene_env`` -- all pure python."""

    def __init__(self, source: str = DEFAULT_SOURCE, scenes_root: Path | str | None = None) -> None:
        self._source = source
        self._scenes_root = Path(scenes_root) if scenes_root is not None else None
        self._embodiment: EmbodimentCfg | None = None
        self.env_spacing: float | None = None
        self.shared_scene = True
        self.robot_pose: tuple[float, float, float] | None = None  # (x, y, yaw_rad) in the env frame

    def configure(
        self,
        env_spacing: float | None = None,
        shared_scene: bool | None = None,
        robot_pose: tuple[float, float, float] | None = None,
    ) -> None:
        """Optional knobs; only arguments that are not None change anything."""
        if env_spacing is not None:
            self.env_spacing = float(env_spacing)
        if shared_scene is not None:
            self.shared_scene = bool(shared_scene)
        if robot_pose is not None:
            self.robot_pose = (float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2]))

    # ---- scenes (pure python) --------------------------------------------------------------------------
    def list_scenes(self) -> list[SceneInfo]:
        return [
            SceneInfo(
                scene_id=s.scene_id,
                category=s.category,
                source=s.source,
                metadata=s.to_metadata(),
            )
            for s in discover_scenes(self._source, self._scenes_root)
        ]

    def normalize_scene_ref(self, ref: str) -> str:
        """``stock`` or ``<source>:<folder>``; Lightwheel kitchens are Arena-only."""
        if ref == STOCK_SCENE_ID or ref.startswith(f"{self._source}:"):
            return ref
        if ref.startswith("lightwheel"):
            raise ValueError(f"plain_usd backend has no Lightwheel scenes ({ref!r}); use --backend arena")
        raise ValueError(f"Unknown scene {ref!r}; use 'stock' or '{self._source}:<folder>'")

    def describe_scene(self, scene_id: str) -> dict[str, Any]:
        """What ``load_scene`` would return (without touching Isaac)."""
        scene_id = self.normalize_scene_ref(scene_id)
        if scene_id == STOCK_SCENE_ID:
            return {"scene_id": scene_id, "kind": "stock", "usd_path": None, "mdl_path": None}
        try:
            files = find_scene(scene_id, self._source, self._scenes_root)
        except FileNotFoundError as e:
            return {"scene_id": scene_id, "kind": self._source, "usd_path": None, "mdl_path": None, "error": str(e)}
        return {
            "scene_id": scene_id,
            "kind": self._source,
            **files.to_metadata(),
            "source_usd_path": str(files.usd_path),
            "usd_path": str(sim_usd_path(files)),  # metres; the raw GRScenes USD is centimetres
        }

    def prepare_scene_env(self, scene_id: str) -> None:
        """Set process env the scene needs; call BEFORE the Isaac app launches (MDL_SYSTEM_PATH)."""
        info = self.describe_scene(scene_id)
        if info.get("mdl_path"):
            ensure_mdl_system_path(info["mdl_path"])

    def load_scene(self, scene_id: str) -> dict[str, Any]:
        """A small dict: ``scene_id``, ``usd_path`` (None for ``stock``), ``mdl_path``, ``variant``. Pure python;
        raises FileNotFoundError if the scene is not on disk."""
        scene_id = self.normalize_scene_ref(scene_id)
        if scene_id == STOCK_SCENE_ID:
            return {"scene_id": scene_id, "usd_path": None, "mdl_path": None, "variant": "stock"}
        files = find_scene(scene_id, self._source, self._scenes_root)
        ensure_mdl_system_path(files.mdl_path)
        return {
            "scene_id": files.scene_id,
            "usd_path": str(sim_usd_path(files)),  # metres; the raw GRScenes USD is centimetres
            "source_usd_path": str(files.usd_path),
            "mdl_path": str(files.mdl_path) if files.mdl_path else None,
            "variant": files.variant,
            "spawn_pose": (0.0, 0.0, 0.0),  # (x, y, yaw) robot start in the scene frame; UNVERIFIED valid per scene
        }

    def register_embodiment(self, cfg: EmbodimentCfg) -> None:
        """Store the cfg; its articulation is resolved lazily by ``build_env``."""
        if "legs" not in cfg.joint_groups:
            raise ValueError(f"Embodiment {cfg.name!r} needs a 'legs' joint group, has {sorted(cfg.joint_groups)}")
        self._embodiment = cfg

    # ---- Isaac Lab side (lazy imports) -----------------------------------------------------------------
    def build_env(self, scene: Any, embodiment: EmbodimentCfg, task: Any, num_envs: int) -> Any:
        """Build and ``gym.make`` a minimal navigation ``ManagerBasedRLEnv``.

        `scene` is the dict from ``load_scene``; `task` is None or a dict (``waypoint_xy``, ``radius``,
        ``episode_length_s``). Legs are driven by name through joint-position actions (zero action = default
        pose); reward = negative distance to the waypoint; termination = timeout or waypoint reached. GPU
        physics: ``sim.device = 'cuda:0'`` (PhysX GPU pipeline follows the device).
        """
        import gymnasium as gym
        import isaaclab.envs.mdp as mdp
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg
        from isaaclab.envs import ManagerBasedRLEnvCfg
        from isaaclab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
        from isaaclab.scene import InteractiveSceneCfg

        try:
            from isaaclab.utils import configclass
        except ImportError:  # pragma: no cover - older layouts
            from isaaclab.utils.configclass import configclass

        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if self._embodiment is None or self._embodiment.name != embodiment.name:
            self.register_embodiment(embodiment)
        params = waypoint_params(task)

        cfgs = make_go2_mdp_cfgs(embodiment)
        cfgs.observations.policy.concatenate_terms = True  # one tensor for the policy group
        robot_cfg = cfgs.robot_cfg
        if self.robot_pose is not None:
            x, y, yaw = self.robot_pose
            robot_cfg = robot_cfg.replace(
                init_state=robot_cfg.init_state.replace(
                    pos=(x, y, DEFAULT_ROBOT_Z), rot=_quat_from_xyzw(_yaw_quat_xyzw(yaw))
                )
            )

        shared = self.shared_scene
        scene_usd = scene.get("usd_path") if isinstance(scene, dict) else None
        if scene_usd:
            position, quat_xyzw = spawn_pose_to_background_pose(tuple(scene.get("spawn_pose", (0.0, 0.0, 0.0))))
            world_cfg = AssetBaseCfg(
                prim_path=SHARED_SCENE_PRIM_PATH if shared else "{ENV_REGEX_NS}/Scene",
                spawn=sim_utils.UsdFileCfg(usd_path=scene_usd),
                init_state=AssetBaseCfg.InitialStateCfg(pos=position, rot=_quat_from_xyzw(quat_xyzw)),
            )
        else:
            world_cfg = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
            shared = True  # the ground plane is always a single global prim
        light_cfg = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        )

        @configclass
        class NavSceneCfg(InteractiveSceneCfg):
            world: AssetBaseCfg = MISSING
            light: AssetBaseCfg = MISSING
            robot: ArticulationCfg = MISSING

        @configclass
        class RewardsCfg:
            waypoint = RewardTermCfg(
                func=waypoint_reward, weight=1.0, params={"waypoint_xy": params["waypoint_xy"]}
            )

        @configclass
        class TerminationsCfg:
            time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
            success = TerminationTermCfg(
                func=waypoint_reached,
                params={"waypoint_xy": params["waypoint_xy"], "radius": params["radius"]},
            )

        @configclass
        class EventsCfg:
            reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")

        spacing = self.env_spacing
        if spacing is None:
            spacing = 0.0 if (shared and scene_usd) else DEFAULT_ENV_SPACING
        scene_cfg = NavSceneCfg(
            num_envs=num_envs,
            env_spacing=spacing,
            world=world_cfg,
            light=light_cfg,
            robot=robot_cfg,
        )
        env_cfg = ManagerBasedRLEnvCfg(
            scene=scene_cfg,
            observations=cfgs.observations,
            actions=cfgs.actions,
            rewards=RewardsCfg(),
            terminations=TerminationsCfg(),
            events=EventsCfg(),
            decimation=DECIMATION,
            episode_length_s=float(params["episode_length_s"]),
        )
        env_cfg.sim.dt = SIM_DT
        env_cfg.sim.render_interval = DECIMATION
        env_cfg.sim.device = "cuda:0"  # GPU physics
        env_cfg.seed = 42

        env_id = f"Cerberus-PlainUsd-Nav-{next(_ENV_IDS)}-v0"
        gym.register(id=env_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True)
        return gym.make(env_id, cfg=env_cfg)
