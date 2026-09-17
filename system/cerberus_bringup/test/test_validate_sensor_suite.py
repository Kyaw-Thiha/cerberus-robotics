import pytest
from cerberus_bringup.validate_sensor_suite import (
    SensorSuiteMismatchError,
    require_explicit_algorithm_if_unset,
    unimplemented_dependencies,
    validate_sensor_requirements,
)


def test_passes_when_all_required_sensors_present():
    validate_sensor_requirements(
        active_sensors={"lidar": True, "front_camera": True, "rear_camera": False, "wrist_camera": False},
        package_requirements={"slam": {"lidar": "required"}},
    )


def test_passes_when_missing_sensor_is_only_optional():
    validate_sensor_requirements(
        active_sensors={"lidar": False, "front_camera": True, "rear_camera": False, "wrist_camera": False},
        package_requirements={"terrain_perception": {"front_camera": "required", "rear_camera": "optional"}},
    )


def test_raises_naming_missing_package_and_sensor():
    with pytest.raises(SensorSuiteMismatchError, match="slam needs lidar"):
        validate_sensor_requirements(
            active_sensors={"lidar": False, "front_camera": True, "rear_camera": False, "wrist_camera": False},
            package_requirements={"slam": {"lidar": "required"}},
        )


def test_raises_listing_every_missing_package_and_sensor():
    with pytest.raises(SensorSuiteMismatchError) as exc_info:
        validate_sensor_requirements(
            active_sensors={"lidar": False, "front_camera": False, "rear_camera": False, "wrist_camera": False},
            package_requirements={
                "slam": {"lidar": "required"},
                "terrain_perception": {"front_camera": "required"},
            },
        )
    assert "slam needs lidar" in str(exc_info.value)
    assert "terrain_perception needs front_camera" in str(exc_info.value)


def test_locomotion_mode_with_no_real_requirement_passes_under_any_profile():
    # blind/perceptive today -- see platforms.yaml -- declare no rig sensor
    # need, so they must pass regardless of the active sensor_profile.
    validate_sensor_requirements(
        active_sensors={"lidar": False, "front_camera": False, "rear_camera": False, "wrist_camera": False},
        package_requirements={"locomotion": {}},
    )


def test_locomotion_mode_with_a_real_sensor_requirement_is_caught_like_any_other_package():
    # Proves the extension point bringup.launch.py now uses (merging a
    # "locomotion" entry into the same package_requirements dict) actually
    # gets validated -- exercised with a synthetic requirement, since no
    # real locomotion mode declares one yet (see platforms.yaml's comment).
    with pytest.raises(SensorSuiteMismatchError, match="locomotion needs front_camera"):
        validate_sensor_requirements(
            active_sensors={"lidar": True, "front_camera": False, "rear_camera": False, "wrist_camera": False},
            package_requirements={
                "locomotion": {"front_camera": "required"},
                "slam": {"lidar": "required"},
            },
        )


def test_unknown_sensor_key_treated_as_absent():
    with pytest.raises(SensorSuiteMismatchError, match="needs wrist_camera"):
        validate_sensor_requirements(
            active_sensors={"lidar": True, "front_camera": True, "rear_camera": False},
            package_requirements={"manipulation": {"wrist_camera": "required"}},
        )


def test_unimplemented_dependencies_returns_empty_when_no_requirements():
    assert unimplemented_dependencies([], {}) == []


def test_unimplemented_dependencies_flags_stub_algorithms():
    assert unimplemented_dependencies(
        ["terrain_perception"], {"terrain_perception": False}
    ) == ["terrain_perception"]


def test_unimplemented_dependencies_passes_real_algorithms():
    assert unimplemented_dependencies(["terrain_perception"], {"terrain_perception": True}) == []


def test_unimplemented_dependencies_treats_unlaunched_package_as_unimplemented():
    # A package that isn't even in implemented_flags (e.g. never launched)
    # is trivially not a real implementation of the dependency.
    assert unimplemented_dependencies(["terrain_perception"], {}) == ["terrain_perception"]


def test_require_explicit_algorithm_returns_value_when_set():
    assert require_explicit_algorithm_if_unset("slam", "fast_lio2") == "fast_lio2"


def test_require_explicit_algorithm_raises_when_none_and_unset():
    with pytest.raises(SensorSuiteMismatchError, match="No recommended algorithm.*'slam'"):
        require_explicit_algorithm_if_unset("slam", None)
