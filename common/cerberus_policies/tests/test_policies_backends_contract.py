"""L1 contract tests for the in-repo backends: load, act() shapes, sensor needs validate against a rig."""

import pytest

from common.cerberus_policies import registry
from common.cerberus_policies.base import Boundary, Role, check_command
from common.cerberus_policies.validate_config import (
    sensor_suite_from_profile,
    validate_config,
)

NAMES = ["null_loco", "scripted_loco"]


@pytest.fixture(autouse=True)
def _clean():
    registry._reset_for_tests()
    yield
    registry._reset_for_tests()


@pytest.mark.parametrize("name", NAMES)
def test_backend_contract(name):
    b = registry.get_backend("loco", name)
    assert b.boundary is Boundary.B1_LOCO and b.role is Role.PRODUCER
    with pytest.raises(RuntimeError):
        b.act({})  # act before load
    with pytest.raises(ValueError):
        b.load("some/checkpoint.pt", None)  # no checkpoint exists for pure-Python sources
    b.load(None, None)
    cmd = b.act({"t": 0.0})
    check_command(Boundary.B1_LOCO, cmd)
    assert isinstance(cmd["vx"], float)


def test_null_is_zero_velocity():
    b = registry.get_backend("loco", "null_loco")
    b.load(None, None)
    cmd = b.act({})
    assert (cmd["vx"], cmd["vy"], cmd["wz"]) == (0.0, 0.0, 0.0)
    assert not cmd["has_body_pitch"] and not cmd["has_body_height"]


def test_scripted_is_constant_from_config():
    b = registry.get_backend("loco", "scripted_loco", config={"vx": 0.5, "wz": -0.2, "body_height": 0.28})
    b.load(None, None)
    first, second = b.act({"t": 0.0}), b.act({"t": 1.0})
    assert first == second
    assert (first["vx"], first["wz"], first["body_height"], first["has_body_height"]) == (0.5, -0.2, 0.28, True)
    first["vx"] = 9.0  # callers cannot corrupt the stored command
    assert b.act({})["vx"] == 0.5


@pytest.mark.parametrize("profile", ["lidar_camera", "camera_only"])
@pytest.mark.parametrize("name", NAMES)
def test_sensor_needs_validate_against_real_rigs(name, profile):
    cls = type(registry.get_backend("loco", name))
    validate_config([cls], sensor_suite_from_profile(profile), external_consumers={Boundary.B1_LOCO: "loco_policy_node"})
