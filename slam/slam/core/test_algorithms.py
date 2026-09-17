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


def test_real_candidates_are_registered():
    names = available_algorithms()
    assert "fast_lio2" in names
    assert "fast_livo2" in names
    assert get_algorithm("fast_lio2").sensor_requirements == {"lidar": "required"}
    assert get_algorithm("fast_livo2").sensor_requirements == {
        "lidar": "required",
        "front_camera": "required",
    }
