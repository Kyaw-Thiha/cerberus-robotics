import pytest

from common.cerberus_policies.base import Boundary, PolicyBackend, Role
from common.cerberus_policies.validate_config import (
    ConfigValidationError,
    sensor_suite_from_profile,
    validate_config,
)

RIG = {"lidar": True, "front_camera": True, "rear_camera": False, "wrist_camera": False, "imu": True}


def _backend(name, boundary, role, sensors=()):
    return type(
        name,
        (PolicyBackend,),
        {
            "boundary": boundary,
            "role": role,
            "required_sensors": tuple(sensors),
            "rate_hz": 50.0,
            "load": lambda self, checkpoint_path, embodiment_cfg: None,
            "act": lambda self, observation: {},
        },
    )


def test_valid_config_one_producer_one_consumer():
    prod = _backend("Planner", Boundary.B1_LOCO, Role.PRODUCER)
    cons = _backend("Policy", Boundary.B1_LOCO, Role.CONSUMER, ["lidar", "joint_encoders"])
    validate_config([prod, cons], RIG)


def test_instances_work_too():
    prod = _backend("Planner", Boundary.B1_LOCO, Role.PRODUCER)
    cons = _backend("Policy", Boundary.B1_LOCO, Role.CONSUMER)
    validate_config([prod(), cons()], RIG)


def test_missing_consumer():
    prod = _backend("Planner", Boundary.B1_LOCO, Role.PRODUCER)
    with pytest.raises(ConfigValidationError, match="B1_LOCO has no consumer"):
        validate_config([prod], RIG)


def test_missing_producer():
    cons = _backend("Policy", Boundary.B2_ARM, Role.CONSUMER)
    with pytest.raises(ConfigValidationError, match="B2_ARM has no producer"):
        validate_config([cons], RIG)


def test_double_producer_and_double_consumer():
    p1, p2 = (_backend(n, Boundary.B1_LOCO, Role.PRODUCER) for n in ("PlannerA", "PlannerB"))
    c1, c2 = (_backend(n, Boundary.B1_LOCO, Role.CONSUMER) for n in ("PolicyA", "PolicyB"))
    with pytest.raises(ConfigValidationError) as e:
        validate_config([p1, p2, c1, c2], RIG)
    assert any("2 producers" in p for p in e.value.problems)
    assert any("2 consumers" in p for p in e.value.problems)


def test_external_endpoints_count():
    cons = _backend("Policy", Boundary.B1_LOCO, Role.CONSUMER)
    validate_config([cons], RIG, external_producers={Boundary.B1_LOCO: "global_planner"})
    prod = _backend("Planner", Boundary.B1_LOCO, Role.PRODUCER)
    with pytest.raises(ConfigValidationError, match="2 producers"):
        validate_config([prod, cons], RIG, external_producers={Boundary.B1_LOCO: "global_planner"})


def test_missing_sensor():
    cons = _backend("Jeplo", Boundary.B1_LOCO, Role.CONSUMER, ["lidar"])
    camera_only = {"lidar": False, "front_camera": True, "rear_camera": True, "wrist_camera": False, "imu": False}
    with pytest.raises(ConfigValidationError, match=r"Jeplo requires sensors .*\['lidar'\]"):
        validate_config([cons], camera_only, external_producers={Boundary.B1_LOCO: "planner"})


def test_all_problems_reported_together():
    cons = _backend("Jeplo", Boundary.B1_LOCO, Role.CONSUMER, ["lidar"])
    with pytest.raises(ConfigValidationError) as e:
        validate_config([cons], {"lidar": False})
    assert len(e.value.problems) == 2  # missing sensor + missing producer


def test_unused_boundaries_are_not_required():
    validate_config([], RIG)


def test_sensor_suite_from_real_profiles():
    assert sensor_suite_from_profile("lidar_camera")["lidar"] is True
    assert sensor_suite_from_profile("camera_only")["lidar"] is False
    with pytest.raises(ValueError, match="Unknown sensor profile"):
        sensor_suite_from_profile("nope")
