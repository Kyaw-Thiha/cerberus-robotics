import pytest

from common.algorithm_registry import AlgorithmCfg
from perception.terrain_perception.core.algorithms import (
    available_algorithms,
    get_algorithm,
    register_algorithm,
)


def test_register_algorithm_adds_to_available_algorithms():
    cfg = AlgorithmCfg(name="regtest1", sensor_requirements={"front_camera": "required"})
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
    with pytest.raises(ValueError, match="unknown terrain understanding algorithm 'nope'"):
        get_algorithm("nope")


def test_real_candidates_are_registered():
    names = available_algorithms()
    assert "ame2" in names
    assert "delta" in names
    assert get_algorithm("ame2").sensor_requirements == {"front_camera": "required"}
    assert get_algorithm("delta").sensor_requirements == {
        "front_camera": "required",
        "rear_camera": "optional",
    }
