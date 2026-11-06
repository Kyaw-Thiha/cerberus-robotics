import pytest

from common.algorithm_registry import AlgorithmCfg, make_algorithm_registry


def test_checkpoints_two_registries_from_same_factory_stay_independent():
    register_a, get_a, available_a = make_algorithm_registry("domain A")
    register_b, get_b, available_b = make_algorithm_registry("domain B")

    register_a(AlgorithmCfg(name="shared_name"))
    register_b(AlgorithmCfg(name="shared_name"))  # must not collide with domain A's

    assert available_a() == ["shared_name"]
    assert available_b() == ["shared_name"]
    assert get_a("shared_name") is not get_b("shared_name")


def test_register_algorithm_rejects_duplicate_name():
    register, _get, _available = make_algorithm_registry("domain")
    register(AlgorithmCfg(name="x"))
    with pytest.raises(ValueError, match="domain algorithm 'x' already registered"):
        register(AlgorithmCfg(name="x"))


def test_get_algorithm_unknown_name_raises_with_domain_and_available_list():
    register, get, _available = make_algorithm_registry("domain")
    register(AlgorithmCfg(name="x"))
    with pytest.raises(ValueError, match="unknown domain algorithm 'nope', available: \\['x'\\]"):
        get("nope")


def test_available_algorithms_sorted():
    register, _get, available = make_algorithm_registry("domain")
    register(AlgorithmCfg(name="zeta"))
    register(AlgorithmCfg(name="alpha"))
    assert available() == ["alpha", "zeta"]


def test_sensor_requirements_and_description_default_empty():
    cfg = AlgorithmCfg(name="x")
    assert cfg.sensor_requirements == {}
    assert cfg.description == ""


def test_implemented_defaults_false():
    # Every one of the 8 real candidates registered today is a declaration
    # only, no real code -- False is the accurate default, not a placeholder.
    assert AlgorithmCfg(name="x").implemented is False
