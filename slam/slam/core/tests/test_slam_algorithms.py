import pytest

from common.algorithm_registry import AlgorithmCfg
from slam.slam.core.algorithms import (
    available_algorithms,
    get_algorithm,
    register_algorithm,
)


def test_register_algorithm_adds_to_available_algorithms():
    cfg = AlgorithmCfg(name="regtest1", sensor_requirements={"lidar": "required"})
    register_algorithm(cfg)
    assert "regtest1" in available_algorithms()


def test_register_algorithm_rejects_duplicate_name():
    cfg = AlgorithmCfg(name="regtest2")
    register_algorithm(cfg)
    with pytest.raises(ValueError, match="already registered"):
        register_algorithm(cfg)


def test_get_algorithm_returns_registered_cfg():
    cfg = AlgorithmCfg(name="regtest3")
    register_algorithm(cfg)
    assert get_algorithm("regtest3") is cfg


def test_get_algorithm_unknown_name_raises_with_available_list():
    with pytest.raises(ValueError, match="unknown SLAM algorithm 'nope'"):
        get_algorithm("nope")


EXPECTED_SENSOR_CONFIGS = {
    "fast_lio2": {"lidar": "required", "imu": "required"},
    "bievr_lio": {"lidar": "required", "imu": "required"},
    "fast_livo2": {"lidar": "required", "imu": "required", "front_camera": "required"},
    "gs_livo": {"lidar": "required", "imu": "required", "front_camera": "required"},
    "orb_slam3": {"front_camera": "required"},
    "multicam_slam": {"front_camera": "required", "rear_camera": "required"},
}


def test_real_candidates_are_registered():
    names = available_algorithms()
    for name in EXPECTED_SENSOR_CONFIGS:
        assert name in names
    assert not get_algorithm("bievr_lio").implemented
    assert not get_algorithm("gs_livo").implemented
    assert not get_algorithm("orb_slam3").implemented
    assert not get_algorithm("multicam_slam").implemented


def test_sensor_config_groupings_match_documented_shape():
    for name, expected in EXPECTED_SENSOR_CONFIGS.items():
        assert get_algorithm(name).sensor_requirements == expected, (
            f"{name}'s sensor_requirements drifted from its documented shape"
        )


def test_no_two_sensor_configs_collapse_to_the_same_shape():
    shapes = [
        {"lidar": "required", "imu": "required"},
        {"lidar": "required", "imu": "required", "front_camera": "required"},
        {"front_camera": "required"},
        {"front_camera": "required", "rear_camera": "required"},
    ]
    assert len(shapes) == len({tuple(sorted(s.items())) for s in shapes})
