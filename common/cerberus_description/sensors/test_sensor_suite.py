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
    }


def test_active_sensors_camera_only_shape():
    cfg = _make_cfg("regtest_camera_only", rear_camera=object())
    assert cfg.active_sensors() == {
        "lidar": False,
        "front_camera": True,
        "rear_camera": True,
        "wrist_camera": False,
    }


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


def test_attach_sensor_suite_camera_only_rig():
    rear_camera_sensor = object()
    suite = SensorSuiteCfg(name="regtest5", lidar=None, front_camera=object(), rear_camera=rear_camera_sensor)
    scene_cfg = _FakeSceneCfg()

    attach_sensor_suite(scene_cfg, suite)

    assert not hasattr(scene_cfg, "lidar")
    assert scene_cfg.rear_camera is rear_camera_sensor
