import pytest

from common.cerberus_description.sensors.sensor_suite import (
    SensorSuiteCfg,
    attach_sensor_suite,
    available_suites,
    get_suite,
    register_suite,
)


def _make_cfg(name: str, *, lidar=None, rear_camera=None) -> SensorSuiteCfg:
    return SensorSuiteCfg(name=name, lidar=lidar, front_camera=object(), rear_camera=rear_camera)


def test_active_sensors_lidar_camera_shape():
    cfg = _make_cfg("regtest_lidar", lidar=object())
    assert cfg.active_sensors() == {
        "lidar": True,
        "front_camera": True,
        "rear_camera": False,
        "wrist_camera": False,
        "imu": False,
    }


def test_active_sensors_camera_only_shape():
    cfg = _make_cfg("regtest_camera_only", rear_camera=object())
    assert cfg.active_sensors() == {
        "lidar": False,
        "front_camera": True,
        "rear_camera": True,
        "wrist_camera": False,
        "imu": False,
    }


def test_active_sensors_imu_present_on_lidar_camera_rig():
    cfg = _make_cfg("regtest_imu", lidar=object())
    object.__setattr__(cfg, "imu", object())
    assert cfg.active_sensors()["imu"] is True


def test_register_suite_adds_to_available_suites():
    cfg = _make_cfg("regtest1")
    register_suite(cfg)
    assert "regtest1" in available_suites()


def test_register_suite_rejects_duplicate_name():
    cfg = _make_cfg("regtest2")
    register_suite(cfg)
    with pytest.raises(ValueError, match="already registered"):
        register_suite(cfg)


def test_get_suite_returns_registered_cfg():
    cfg = _make_cfg("regtest3")
    register_suite(cfg)
    assert get_suite("regtest3") is cfg


def test_get_suite_unknown_name_raises_with_available_list():
    with pytest.raises(ValueError, match="unknown sensor suite 'nope'"):
        get_suite("nope")


class _FakeSceneCfg:
    """Stands in for an Isaac Lab @configclass scene cfg -- a plain mutable
    object, no isaaclab dependency needed to test attribute assignment."""


def test_attach_sensor_suite_sets_only_populated_slots():
    lidar_sensor = object()
    front_camera_sensor = object()
    suite = SensorSuiteCfg(name="regtest4", lidar=lidar_sensor, front_camera=front_camera_sensor, rear_camera=None)
    scene_cfg = _FakeSceneCfg()

    attach_sensor_suite(scene_cfg, suite)

    assert scene_cfg.lidar is lidar_sensor
    assert scene_cfg.front_camera is front_camera_sensor
    assert not hasattr(scene_cfg, "rear_camera")
    assert not hasattr(scene_cfg, "wrist_camera")
    assert not hasattr(scene_cfg, "imu")


def test_attach_sensor_suite_wires_imu_when_present():
    imu_sensor = object()
    suite = SensorSuiteCfg(name="regtest_imu_attach", lidar=object(), front_camera=object(), rear_camera=None, imu=imu_sensor)
    scene_cfg = _FakeSceneCfg()

    attach_sensor_suite(scene_cfg, suite)

    assert scene_cfg.imu is imu_sensor


def test_attach_sensor_suite_camera_only_rig():
    rear_camera_sensor = object()
    suite = SensorSuiteCfg(name="regtest5", lidar=None, front_camera=object(), rear_camera=rear_camera_sensor)
    scene_cfg = _FakeSceneCfg()

    attach_sensor_suite(scene_cfg, suite)

    assert not hasattr(scene_cfg, "lidar")
    assert scene_cfg.rear_camera is rear_camera_sensor


def test_sensor_suite_cfg_has_showcase_camera_field_defaulting_to_none():
    from common.cerberus_description.sensors.sensor_suite import SensorSuiteCfg
    suite = SensorSuiteCfg(name="regtest6", lidar=None, front_camera=object(), rear_camera=None, wrist_camera=None)
    assert suite.showcase_camera is None


def test_locked_suites_do_not_spawn_showcase_camera_by_default():
    pytest.importorskip("isaaclab")  # sensor_configs.py imports isaaclab, only verifiable on a pod
    from common.cerberus_description.sensors.sensor_configs import (
        CAMERA_ONLY_SUITE,
        LIDAR_CAMERA_SUITE,
    )
    assert LIDAR_CAMERA_SUITE.showcase_camera is None
    assert CAMERA_ONLY_SUITE.showcase_camera is None


def test_with_showcase_camera_attaches_without_mutating_the_original():
    pytest.importorskip("isaaclab")  # sensor_configs.py imports isaaclab, only verifiable on a pod
    from common.cerberus_description.sensors.sensor_configs import (
        LIDAR_CAMERA_SUITE,
        SHOWCASE_CAMERA_CFG,
    )
    from common.cerberus_description.sensors.sensor_suite import with_showcase_camera
    eval_suite = with_showcase_camera(LIDAR_CAMERA_SUITE, SHOWCASE_CAMERA_CFG)
    assert eval_suite.showcase_camera is SHOWCASE_CAMERA_CFG
    assert LIDAR_CAMERA_SUITE.showcase_camera is None  # original untouched
    assert eval_suite.lidar is LIDAR_CAMERA_SUITE.lidar  # every other field carried over


def test_sensor_suite_cfg_stereo_right_fields_default_to_none():
    # lidar_camera/camera_only never populate these -- only a genuine
    # stereo-pair rig (camera_stereo, see sensor_configs.py) does.
    suite = SensorSuiteCfg(name="regtest7", lidar=None, front_camera=object(), rear_camera=None)
    assert suite.front_camera_right is None
    assert suite.rear_camera_right is None


def test_active_sensors_excludes_stereo_right_fields():
    # front_camera_right/rear_camera_right are a presence-only detail of HOW
    # front_camera/rear_camera got populated, not a distinct modality any
    # sensor_requirements manifest gates on -- active_sensors()'s shape must
    # stay exactly 4 keys regardless of whether a stereo pair is wired.
    cfg = _make_cfg("regtest8", rear_camera=object())
    object.__setattr__(cfg, "front_camera_right", object())
    object.__setattr__(cfg, "rear_camera_right", object())
    assert set(cfg.active_sensors()) == {"lidar", "front_camera", "rear_camera", "wrist_camera", "imu"}


def test_attach_sensor_suite_wires_stereo_right_cameras_when_present():
    front_right = object()
    rear_right = object()
    suite = SensorSuiteCfg(
        name="regtest9",
        lidar=None,
        front_camera=object(),
        front_camera_right=front_right,
        rear_camera=object(),
        rear_camera_right=rear_right,
    )
    scene_cfg = _FakeSceneCfg()

    attach_sensor_suite(scene_cfg, suite)

    assert scene_cfg.front_camera_right is front_right
    assert scene_cfg.rear_camera_right is rear_right


def test_attach_sensor_suite_skips_stereo_right_cameras_when_absent():
    suite = _make_cfg("regtest10")
    scene_cfg = _FakeSceneCfg()

    attach_sensor_suite(scene_cfg, suite)

    assert not hasattr(scene_cfg, "front_camera_right")
    assert not hasattr(scene_cfg, "rear_camera_right")


def test_camera_stereo_suite_wires_both_stereo_pairs_rgb_only_on_right():
    pytest.importorskip("isaaclab")  # sensor_configs.py imports isaaclab, only verifiable on a pod
    from common.cerberus_description.sensors.sensor_configs import (
        CAMERA_STEREO_SUITE,
        FRONT_CAMERA_LEFT_CFG,
        FRONT_CAMERA_RIGHT_CFG,
        REAR_CAMERA_LEFT_CFG,
        REAR_CAMERA_RIGHT_CFG,
    )
    assert CAMERA_STEREO_SUITE.front_camera is FRONT_CAMERA_LEFT_CFG
    assert CAMERA_STEREO_SUITE.front_camera_right is FRONT_CAMERA_RIGHT_CFG
    assert CAMERA_STEREO_SUITE.rear_camera is REAR_CAMERA_LEFT_CFG
    assert CAMERA_STEREO_SUITE.rear_camera_right is REAR_CAMERA_RIGHT_CFG
    # coarse presence check (what validate_sensor_suite.py actually gates on)
    # is unaffected by the extra stereo wiring
    assert CAMERA_STEREO_SUITE.active_sensors() == {
        "lidar": False,
        "front_camera": True,
        "rear_camera": True,
        "wrist_camera": False,
        "imu": False,
    }
    # multicam_slam's actual depth source: RIGHT cameras are rgb-only (no
    # privileged depth render cost for a buffer nothing reads); LEFT keeps
    # rgb+depth as an eval baseline only, never consumed by the SLAM pipeline.
    assert FRONT_CAMERA_RIGHT_CFG.data_types == ["rgb"]
    assert REAR_CAMERA_RIGHT_CFG.data_types == ["rgb"]
    assert FRONT_CAMERA_LEFT_CFG.data_types == ["rgb", "depth"]
    assert REAR_CAMERA_LEFT_CFG.data_types == ["rgb", "depth"]


def test_camera_only_suite_still_single_camera_per_position_unchanged():
    # orb_slam3's rig must stay untouched by the multicam_slam stereo work.
    pytest.importorskip("isaaclab")
    from common.cerberus_description.sensors.sensor_configs import CAMERA_ONLY_SUITE
    assert CAMERA_ONLY_SUITE.front_camera_right is None
    assert CAMERA_ONLY_SUITE.rear_camera_right is None


def test_lidar_camera_suite_has_imu_camera_only_rigs_do_not():
    # fast_lio2/fast_livo2/bievr_lio/gs_livo are lidar-inertial, not
    # lidar-only -- see slam/slam/core/algorithms.py's sensor_requirements.
    # orb_slam3/multicam_slam declare no IMU requirement.
    pytest.importorskip("isaaclab")
    from common.cerberus_description.sensors.sensor_configs import (
        CAMERA_ONLY_SUITE,
        CAMERA_STEREO_SUITE,
        IMU_CFG,
        LIDAR_CAMERA_SUITE,
    )
    assert LIDAR_CAMERA_SUITE.imu is IMU_CFG
    assert CAMERA_ONLY_SUITE.imu is None
    assert CAMERA_STEREO_SUITE.imu is None


def test_camera_offset_rot_literals_convert_per_convention(monkeypatch):
    # Pure-python: the w-first Cerberus literals route through to_isaac. Lab 2.x wants
    # (w, x, y, z); Lab 3.x (camera_cfg.py OffsetCfg.rot) wants (x, y, z, w).
    from common.cerberus_compat.quat import to_isaac

    front_wxyz, rear_wxyz = (0.5, -0.5, 0.5, -0.5), (-0.5, 0.5, 0.5, -0.5)
    monkeypatch.setenv("CERBERUS_QUAT_CONVENTION", "wxyz")
    assert to_isaac(front_wxyz) == front_wxyz
    monkeypatch.setenv("CERBERUS_QUAT_CONVENTION", "xyzw")
    assert to_isaac(front_wxyz) == (-0.5, 0.5, -0.5, 0.5)
    assert to_isaac(rear_wxyz) == (0.5, 0.5, -0.5, -0.5)


@pytest.mark.parametrize(
    "convention, front, rear",
    [
        ("wxyz", (0.5, -0.5, 0.5, -0.5), (-0.5, 0.5, 0.5, -0.5)),
        ("xyzw", (-0.5, 0.5, -0.5, 0.5), (0.5, 0.5, -0.5, -0.5)),
    ],
)
def test_sensor_configs_camera_offsets_follow_convention(convention, front, rear):
    pytest.importorskip("isaaclab")
    import os
    import subprocess
    import sys

    code = (
        "from common.cerberus_description.sensors.sensor_configs import "
        "FRONT_CAMERA_LEFT_CFG as f, REAR_CAMERA_RIGHT_CFG as r; "
        "print(tuple(f.offset.rot), tuple(r.offset.rot))"
    )
    env = dict(os.environ, CERBERUS_QUAT_CONVENTION=convention)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    assert out.strip() == f"{front} {rear}"
