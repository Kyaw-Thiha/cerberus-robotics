import dataclasses
import sys
import types
from types import SimpleNamespace

import pytest

from common.cerberus_worlds.admission import SpawnPose
from slam.slam.eval import scene_env_cfg as sec


@dataclasses.dataclass
class _Replaceable:
    def replace(self, **changes):
        return dataclasses.replace(self, **changes)


@dataclasses.dataclass
class _InitState(_Replaceable):
    pos: tuple = (0.0, 0.0, 0.4)
    rot: tuple = (0.0, 0.0, 0.0, 1.0)


@dataclasses.dataclass
class _Robot(_Replaceable):
    init_state: _InitState = dataclasses.field(default_factory=_InitState)


@dataclasses.dataclass
class _Offset:
    pos: tuple = (0.0, 0.0, 0.05)


@dataclasses.dataclass
class _RayCasterCfg:
    class_type: str = "RayCaster"
    prim_path: str = "{ENV_REGEX_NS}/Robot/base"
    update_period: float = 0.1
    offset: _Offset = dataclasses.field(default_factory=_Offset)
    ray_alignment: str = "yaw"
    mesh_prim_paths: list = dataclasses.field(default_factory=lambda: ["/World/ground"])
    pattern_cfg: str = "lidar-pattern"


@dataclasses.dataclass
class _RaycastTargetCfg:
    prim_expr: str
    merge_prim_meshes: bool = True
    track_mesh_transforms: bool = True


@dataclasses.dataclass
class _MultiMeshRayCasterCfg(_RayCasterCfg):
    class_type: str = "MultiMeshRayCaster"
    RaycastTargetCfg = _RaycastTargetCfg


@dataclasses.dataclass
class _TerrainImporterCfg:
    prim_path: str
    terrain_type: str
    usd_path: str
    collision_group: int
    debug_vis: bool


@dataclasses.dataclass
class _AssetBaseCfg:
    prim_path: str
    spawn: object


@pytest.fixture
def fake_isaaclab(monkeypatch):
    """Light stand-ins for the four isaaclab classes apply_scene builds (isaaclab is not installed here)."""
    sim = types.ModuleType("isaaclab.sim")
    sim.DomeLightCfg = lambda **kw: ("dome", kw)
    sim.UsdFileCfg = lambda **kw: ("usd", kw)
    modules = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.sim": sim,
        "isaaclab.assets": types.SimpleNamespace(AssetBaseCfg=_AssetBaseCfg),
        "isaaclab.sensors": types.SimpleNamespace(MultiMeshRayCasterCfg=_MultiMeshRayCasterCfg),
        "isaaclab.terrains": types.SimpleNamespace(TerrainImporterCfg=_TerrainImporterCfg),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("CERBERUS_QUAT_CONVENTION", "xyzw")  # Lab 3.0 native order


def _env_cfg(lidar=True):
    scene = SimpleNamespace(
        num_envs=50,
        terrain="generated-ground",
        height_scanner=None,
        sky_light="stock-sky",
        robot=_Robot(),
        lidar=_RayCasterCfg() if lidar else None,
    )
    reset_base = SimpleNamespace(params={"pose_range": {"x": (-0.5, 0.5)}, "velocity_range": {"x": (-0.5, 0.5)}})
    return SimpleNamespace(
        scene=scene,
        curriculum=SimpleNamespace(terrain_levels=None),
        episode_length_s=20.0,
        commands=SimpleNamespace(
            base_velocity=SimpleNamespace(
                heading_command=True, rel_standing_envs=0.02, resampling_time_range=(10.0, 10.0), debug_vis=True
            )
        ),
        events=SimpleNamespace(
            reset_base=reset_base, reset_robot_joints=SimpleNamespace(params={"position_range": (0.5, 1.5)})
        ),
        sim=SimpleNamespace(physics=type("PhysxCfg", (), {})(), use_newton_actuators=False),
    )


@pytest.fixture
def scene(tmp_path):
    """A scene folder that already has its LiDAR proxy (scripts/build_lidar_proxy.py output)."""
    (tmp_path / "lidar_proxy.usd").write_text("")
    return {"scene_id": "grscenes:a", "usd_path": str(tmp_path / "start_result_navigation.usd"), "spawn_pose": (0.0, 0.0, 0.0)}


def test_apply_scene_replaces_ground_with_scene_usd(fake_isaaclab, scene):
    cfg = _env_cfg()
    sec.apply_scene(cfg, scene, SpawnPose(1.0, 2.0, 0.0, 0.1))
    terrain = cfg.scene.terrain
    assert (terrain.prim_path, terrain.terrain_type, terrain.usd_path) == ("/World/ground", "usd", scene["usd_path"])
    assert terrain.collision_group == -1 and terrain.debug_vis is False
    assert terrain.usd_path == scene["usd_path"]
    assert cfg.scene.sky_light.spawn[0] == "dome"


def test_apply_scene_places_robot_at_floor_plus_standing_height(fake_isaaclab, scene):
    cfg = _env_cfg()
    stock_robot = cfg.scene.robot
    sec.apply_scene(cfg, scene, SpawnPose(1.0, 2.0, 0.0, 0.1))
    assert cfg.scene.robot.init_state.pos == pytest.approx((1.0, 2.0, 0.5))  # floor 0.1 + stock init z 0.4
    assert cfg.scene.robot.init_state.rot == pytest.approx((0.0, 0.0, 0.0, 1.0))  # xyzw identity
    assert stock_robot.init_state.pos == (0.0, 0.0, 0.4)  # the shared source cfg is not mutated


def test_apply_scene_yaw_quaternion_is_xyzw_on_lab3(fake_isaaclab, scene):
    cfg = _env_cfg()
    sec.apply_scene(cfg, scene, SpawnPose(0.0, 0.0, 3.141592653589793 / 2, 0.0))
    x, y, z, w = cfg.scene.robot.init_state.rot
    assert (x, y) == (0.0, 0.0)
    assert z == pytest.approx(0.7071067811865476) and w == pytest.approx(0.7071067811865476)


def test_apply_scene_swaps_lidar_for_multimesh_keeping_semantics(fake_isaaclab, scene):
    cfg = _env_cfg()
    stock = cfg.scene.lidar
    sec.apply_scene(cfg, scene, SpawnPose(0.0, 0.0, 0.0, 0.0))
    lidar = cfg.scene.lidar
    assert isinstance(lidar, _MultiMeshRayCasterCfg)
    assert (lidar.update_period, lidar.offset, lidar.ray_alignment, lidar.pattern_cfg, lidar.prim_path) == (
        stock.update_period, stock.offset, stock.ray_alignment, stock.pattern_cfg, stock.prim_path
    )
    (target,) = lidar.mesh_prim_paths
    assert target == _RaycastTargetCfg("/World/lidar_proxy", merge_prim_meshes=True, track_mesh_transforms=False)
    assert stock.mesh_prim_paths == ["/World/ground"]  # LIDAR_CFG itself is untouched


def test_apply_scene_spawns_the_proxy_invisible_at_the_target_path(fake_isaaclab, scene, tmp_path):
    cfg = _env_cfg()
    sec.apply_scene(cfg, scene, SpawnPose(0.0, 0.0, 0.0, 0.0))
    proxy = cfg.scene.lidar_proxy
    assert proxy.prim_path == sec.LIDAR_PROXY_PRIM_PATH == cfg.scene.lidar.mesh_prim_paths[0].prim_expr
    assert proxy.spawn == ("usd", {"usd_path": str(tmp_path / "lidar_proxy.usd"), "visible": False})


def test_apply_scene_requires_the_proxy_for_a_lidar_rig(fake_isaaclab, tmp_path):
    scene = {"scene_id": "grscenes:a", "usd_path": str(tmp_path / "start_result_navigation.usd")}
    with pytest.raises(FileNotFoundError, match="build_lidar_proxy.py --scene grscenes:a"):
        sec.apply_scene(_env_cfg(), scene, SpawnPose(0.0, 0.0, 0.0, 0.0))
    camera_only = _env_cfg(lidar=False)  # cameras do not ray-cast: no proxy needed, none spawned
    sec.apply_scene(camera_only, scene, SpawnPose(0.0, 0.0, 0.0, 0.0))
    assert not hasattr(camera_only.scene, "lidar_proxy")


def test_apply_scene_camera_only_rig_has_no_lidar_to_swap(fake_isaaclab, scene):
    cfg = _env_cfg(lidar=False)
    sec.apply_scene(cfg, scene, SpawnPose(0.0, 0.0, 0.0, 0.0))
    assert cfg.scene.lidar is None


def test_apply_scene_rejects_stock_scene_dict(fake_isaaclab):
    with pytest.raises(ValueError, match="no usd_path"):
        sec.apply_scene(_env_cfg(), {"scene_id": "stock", "usd_path": None}, SpawnPose(0.0, 0.0, 0.0, 0.0))


def test_apply_scene_rejects_cfg_that_still_uses_the_generator_terrain(fake_isaaclab, scene):
    cfg = _env_cfg()
    cfg.scene.height_scanner = object()
    with pytest.raises(ValueError, match="height_scanner"):
        sec.apply_scene(cfg, scene, SpawnPose(0.0, 0.0, 0.0, 0.0))
    cfg = _env_cfg()
    cfg.curriculum.terrain_levels = object()
    with pytest.raises(ValueError, match="terrain_levels"):
        sec.apply_scene(cfg, scene, SpawnPose(0.0, 0.0, 0.0, 0.0))


def test_apply_mission_makes_the_command_and_start_deterministic():
    cfg = _env_cfg()
    sec.apply_mission(cfg, 21.0)
    assert cfg.scene.num_envs == 1
    assert cfg.episode_length_s == pytest.approx(21.0 + sec.EPISODE_MARGIN_S)
    command = cfg.commands.base_velocity
    assert command.heading_command is False and command.rel_standing_envs == 0.0 and command.debug_vis is False
    assert min(command.resampling_time_range) > 1e6
    pose = cfg.events.reset_base.params["pose_range"]
    assert pose == {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    assert set(cfg.events.reset_base.params["velocity_range"]) == {"x", "y", "z", "roll", "pitch", "yaw"}
    assert all(v == (0.0, 0.0) for v in cfg.events.reset_base.params["velocity_range"].values())
    assert cfg.events.reset_robot_joints.params["position_range"] == (1.0, 1.0)


def test_episode_outlasts_the_household_mission():
    from slam.slam.eval.scripted_motion import canned_schedule

    mission_s = sum(s.duration_seconds for s in canned_schedule("household")) + 2.0
    assert 20.0 < mission_s  # the stock 20 s episode would reset the robot mid-mission
    cfg = _env_cfg()
    sec.apply_mission(cfg, mission_s)
    assert cfg.episode_length_s > mission_s


def test_physics_summary_accepts_physx_with_legacy_actuators():
    assert sec.physics_summary(_env_cfg()) == {"physics_cfg": "PhysxCfg", "use_newton_actuators": False}


def test_physics_summary_rejects_newton_and_newton_actuators():
    cfg = _env_cfg()
    cfg.sim.physics = type("NewtonCfg", (), {})()
    with pytest.raises(RuntimeError, match="PhysX"):
        sec.physics_summary(cfg)
    cfg = _env_cfg()
    cfg.sim.use_newton_actuators = True
    with pytest.raises(RuntimeError, match="legacy actuator"):
        sec.physics_summary(cfg)


def test_physics_summary_lab2_single_backend():
    cfg = _env_cfg()
    cfg.sim = SimpleNamespace()
    assert sec.physics_summary(cfg)["use_newton_actuators"] is False
