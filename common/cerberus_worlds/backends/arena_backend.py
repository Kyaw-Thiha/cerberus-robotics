"""Isaac Lab Arena world backend.

Composes Scene + embodiment + task through Arena's ``ArenaEnvBuilder`` into an Isaac Lab
``ManagerBasedRLEnv``. Any isaaclab/arena import stays inside method bodies so importing this module works
without the sim installed (the registry imports it lazily, and ``list_scenes`` / ``describe_scene`` /
``normalize_scene_ref`` are pure python).

Arena API used (third_party/IsaacLab-Arena at ``main``, see ``arena_assets/go2_embodiment.py`` for the full notes):
  * ``isaaclab_arena.assets.registries.AssetRegistry().get_asset_by_name(name)()`` -> Background / Embodiment
  * ``isaaclab_arena.scene.scene.Scene(assets=[...])``
  * ``isaaclab_arena.environments.isaaclab_arena_environment.IsaacLabArenaEnvironment(name, scene, embodiment, task)``
  * ``isaaclab_arena.environments.arena_env_builder_cfg.ArenaEnvBuilderCfg(num_envs, env_spacing, ...)`` and
    ``isaaclab_arena.environments.arena_env_builder.ArenaEnvBuilder(arena_env, cfg).make_registered()``

Scene ids: ``lightwheel_kitchen_<layout>_<style>`` (100 Lightwheel RoboCasa kitchens, USD downloaded through
``lightwheel_sdk`` at load time; the 2026-10-05 check in docs/arena_integration_log.md found anonymous access
works for layout 1 / style 1 only), ``grscenes:<folder>`` (plain GRScenes USD under ``data/scenes/grscenes``,
see ``arena_assets``), and ``stock`` (an Arena library background that needs no extra data; spike control).

Quaternions are XYZW on the Lab 3.0 line (Arena ``Pose.rotation_xyzw``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from common.cerberus_description.embodiment import EmbodimentCfg
from common.cerberus_worlds.arena_assets import (
    discover_scenes,
    ensure_mdl_system_path,
    sim_usd_path,
    find_scene,
)
from common.cerberus_worlds.base import SceneInfo, WorldBackend

# Copied from IsaacLab-Arena/isaaclab_arena/assets/lightwheel_kitchen_factory.py (main @ 311d0f0fc), which in turn
# cites https://docs.lightwheel.net/lw_benchhub/task%20suites/Lightwheel%20Robocasa%20Tasks/Scenes/Kitchen%20Scenes
# Entries are (id, ClassName fragment, snake_case name); Arena registers one asset per (layout, style) pair named
# "lightwheel_kitchen_<layout snake_case>_<style snake_case>" with layout_id / style_id class attributes.
LIGHTWHEEL_KITCHEN_LAYOUTS: tuple[tuple[int, str, str], ...] = (
    (1, "OneWall", "one_wall"),
    (2, "OneWallWithIsland", "one_wall_with_island"),
    (3, "LShaped", "l_shaped"),
    (4, "LShapedWithIsland", "l_shaped_with_island"),
    (5, "Galley", "galley"),
    (6, "UShaped", "u_shaped"),
    (7, "UShapedWithIsland", "u_shaped_with_island"),
    (8, "GShaped", "g_shaped"),
    (9, "GShapedLarge", "g_shaped_large"),
    (10, "Wraparound", "wraparound"),
)
LIGHTWHEEL_KITCHEN_STYLES: tuple[tuple[int, str, str], ...] = (
    (1, "Coastal", "coastal"),
    (2, "Farmhouse1", "farmhouse1"),
    (3, "Industrial", "industrial"),
    (4, "Mediterranean", "mediterranean"),
    (5, "Modern1", "modern1"),
    (6, "Modern2", "modern2"),
    (7, "Rustic", "rustic"),
    (8, "Scandinavian", "scandinavian"),
    (9, "Traditional", "traditional"),
    (10, "Farmhouse2", "farmhouse2"),
)

KITCHEN_PREFIX = "lightwheel_kitchen_"
STOCK_SCENE_ID = "stock"
STOCK_BACKGROUND = "galileo"  # Arena library background (assets/background_library.py); # UNVERIFIED as a Go2 floor
LIGHT_ASSET = "light"  # Arena library dome light (docs: external_environments.rst)
GRSCENES_SOURCE = "grscenes"
DEFAULT_ROBOT_Z = 0.4  # UNITREE_GO2_CFG init_state.pos z


def kitchen_scene_id(layout_id: int, style_id: int) -> str:
    """Arena asset name of the kitchen with 1-based `layout_id` / `style_id`."""
    layouts = {i: name for i, _, name in LIGHTWHEEL_KITCHEN_LAYOUTS}
    styles = {i: name for i, _, name in LIGHTWHEEL_KITCHEN_STYLES}
    if layout_id not in layouts or style_id not in styles:
        raise ValueError(f"Lightwheel kitchen layout must be 1..10 and style 1..10, got {layout_id}:{style_id}")
    return f"{KITCHEN_PREFIX}{layouts[layout_id]}_{styles[style_id]}"


def kitchen_scene_ids() -> list[str]:
    """All 100 kitchen asset names (layout-major order)."""
    return [
        f"{KITCHEN_PREFIX}{layout}_{style}"
        for _, _, layout in LIGHTWHEEL_KITCHEN_LAYOUTS
        for _, _, style in LIGHTWHEEL_KITCHEN_STYLES
    ]


def _kitchen_infos() -> list[SceneInfo]:
    infos = []
    for layout_id, _, layout in LIGHTWHEEL_KITCHEN_LAYOUTS:
        for style_id, _, style in LIGHTWHEEL_KITCHEN_STYLES:
            infos.append(
                SceneInfo(
                    scene_id=f"{KITCHEN_PREFIX}{layout}_{style}",
                    category="kitchen",
                    source="lightwheel",
                    metadata={"layout_id": layout_id, "layout": layout, "style_id": style_id, "style": style},
                )
            )
    return infos


class ArenaBackend(WorldBackend):
    """Arena-backed worlds. Extra (non-abstract) helpers: ``configure``, ``normalize_scene_ref``,
    ``describe_scene`` -- all pure python."""

    def __init__(self, scenes_root: Path | str | None = None) -> None:
        self._scenes_root = Path(scenes_root) if scenes_root is not None else None
        self._embodiments: dict[str, EmbodimentCfg] = {}
        self._arena_names: dict[str, str] = {}
        self._env_counter = 0
        self.env_spacing: float | None = None  # None: Arena default (30 m), or 0 with a shared scene
        self.shared_scene = False  # GRScenes: spawn once at /World/Scene instead of once per env
        self.robot_pose: tuple[float, float, float] | None = None  # (x, y, yaw_rad) in the env frame

    # ---- configuration -------------------------------------------------------------------------------
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
        infos = _kitchen_infos()
        for scene in discover_scenes(GRSCENES_SOURCE, self._scenes_root):
            infos.append(
                SceneInfo(
                    scene_id=scene.scene_id,
                    category=scene.category or "home",
                    source="grscenes",
                    metadata=scene.to_metadata(),
                )
            )
        return infos

    def normalize_scene_ref(self, ref: str) -> str:
        """Map a CLI-style ref to a scene id: ``stock``, ``grscenes:<folder>``, ``lightwheel:<layout>:<style>``
        (1-based ints) or an exact kitchen id. Raises ValueError for anything else."""
        if ref == STOCK_SCENE_ID:
            return ref
        if ref.startswith("lightwheel:"):
            parts = ref.split(":")
            try:
                if len(parts) != 3:
                    raise ValueError
                return kitchen_scene_id(int(parts[1]), int(parts[2]))
            except ValueError:
                raise ValueError(f"Bad Lightwheel ref {ref!r}; expected lightwheel:<layout 1-10>:<style 1-10>") from None
        if ref.startswith("grscenes:"):
            return ref
        if ref in kitchen_scene_ids():
            return ref
        raise ValueError(
            f"Unknown scene {ref!r}; use 'stock', 'grscenes:<folder>', 'lightwheel:<layout>:<style>' or a "
            f"{KITCHEN_PREFIX}<layout>_<style> id"
        )

    def describe_scene(self, scene_id: str) -> dict[str, Any]:
        """What ``load_scene`` would build, without touching Arena (for ``--dry-run`` and logging)."""
        scene_id = self.normalize_scene_ref(scene_id)
        if scene_id == STOCK_SCENE_ID:
            return {"scene_id": scene_id, "kind": "stock", "arena_asset": STOCK_BACKGROUND}
        if scene_id.startswith("grscenes:"):
            try:
                files = find_scene(scene_id, GRSCENES_SOURCE, self._scenes_root)
            except FileNotFoundError as e:
                return {"scene_id": scene_id, "kind": "grscenes", "usd_path": None, "error": str(e)}
            return {
                "scene_id": scene_id,
                "kind": "grscenes",
                **files.to_metadata(),
                "source_usd_path": str(files.usd_path),
                "usd_path": str(sim_usd_path(files)),  # metres; the raw GRScenes USD is centimetres
            }
        return {"scene_id": scene_id, "kind": "lightwheel", "arena_asset": scene_id}

    def prepare_scene_env(self, scene_id: str) -> None:
        """Set process env the scene needs; call BEFORE the Isaac app launches (MDL_SYSTEM_PATH for GRScenes)."""
        info = self.describe_scene(scene_id)
        if info.get("kind") == "grscenes" and info.get("mdl_path"):
            ensure_mdl_system_path(info["mdl_path"])

    # ---- Arena-side (lazy imports) ---------------------------------------------------------------------
    def load_scene(self, scene_id: str) -> Any:
        """Build an Arena ``Scene`` (background + light). Needs the Isaac app running with Arena installed."""
        from isaaclab_arena.assets.registries import AssetRegistry
        from isaaclab_arena.scene.scene import Scene

        from common.cerberus_worlds.arena_assets.go2_embodiment import get_arena_classes

        scene_id = self.normalize_scene_ref(scene_id)
        registry = AssetRegistry()
        if scene_id.startswith("grscenes:"):
            files = find_scene(scene_id, GRSCENES_SOURCE, self._scenes_root)  # raises FileNotFoundError
            ensure_mdl_system_path(files.mdl_path)
            background = get_arena_classes()["GRScenesBackground"](
                usd_path=str(sim_usd_path(files)),  # metres; the raw GRScenes USD is centimetres
                spawn_pose=(0.0, 0.0, 0.0),  # UNVERIFIED: GRScenes origin is a valid floor spot; tune per scene
                mdl_path=str(files.mdl_path) if files.mdl_path else None,
                shared=self.shared_scene,
            )
        else:
            asset = STOCK_BACKGROUND if scene_id == STOCK_SCENE_ID else scene_id
            background = registry.get_asset_by_name(asset)()
        light = registry.get_asset_by_name(LIGHT_ASSET)()
        return Scene(assets=[background, light])

    def register_embodiment(self, cfg: EmbodimentCfg) -> None:
        """Register `cfg`'s Arena embodiment: ``go2`` (stock Go2) or ``go2_arx5`` (fixed-base Go2 + ARX5, W2b).

        Anything else raises ValueError. Needs the Isaac app running (Arena imports pxr/Kit modules).
        """
        from common.cerberus_worlds.arena_assets.go2_embodiment import GO2_ARX5_ARENA_NAME, register_embodiment_cfg

        self._arena_names[cfg.name] = register_embodiment_cfg(cfg)
        if cfg.name == "go2_arx5":
            assert self._arena_names[cfg.name] == GO2_ARX5_ARENA_NAME
        self._embodiments[cfg.name] = cfg

    def build_env(self, scene: Any, embodiment: EmbodimentCfg, task: Any, num_envs: int) -> Any:
        """Compose scene + embodiment + task with ``ArenaEnvBuilder`` and return the gym env.

        `task`: None (default ``WaypointReachTask``), a dict of its kwargs, or an Arena ``TaskBase`` instance.
        `num_envs` goes through ``ArenaEnvBuilderCfg(num_envs=...)``; device/physics are the builder defaults
        (``cuda:0``, PhysX, GPU pipeline). ``env_spacing`` comes from ``configure()``.
        """
        from isaaclab_arena.assets.registries import AssetRegistry
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.utils.pose import Pose

        from common.cerberus_worlds.arena_assets.go2_embodiment import get_arena_classes

        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if embodiment.name not in self._arena_names:
            self.register_embodiment(embodiment)

        initial_pose = None
        if self.robot_pose is not None:
            x, y, yaw = self.robot_pose
            initial_pose = Pose(
                position_xyz=(x, y, DEFAULT_ROBOT_Z),
                rotation_xyzw=(0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)),
            )
        arena_embodiment = AssetRegistry().get_asset_by_name(self._arena_names[embodiment.name])(
            initial_pose=initial_pose
        )

        if task is None:
            task = get_arena_classes()["WaypointReachTask"]()
        elif isinstance(task, dict):
            task = get_arena_classes()["WaypointReachTask"](**task)

        self._env_counter += 1
        arena_env = IsaacLabArenaEnvironment(
            name=f"cerberus_arena_{embodiment.name}_{self._env_counter}",
            scene=scene,
            embodiment=arena_embodiment,
            task=task,
        )
        cfg_kwargs: dict[str, Any] = {"num_envs": num_envs, "solve_relations": False}
        spacing = self.env_spacing
        if spacing is None and self.shared_scene:
            spacing = 0.0  # one shared scene: all robots start at the spawn point, envs isolated by collision groups
        if spacing is not None:
            cfg_kwargs["env_spacing"] = spacing
        builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(**cfg_kwargs))
        return builder.make_registered()
