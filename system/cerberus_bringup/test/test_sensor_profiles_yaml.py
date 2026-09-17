from pathlib import Path

import pytest
import yaml

SENSOR_PROFILES_YAML = Path(__file__).resolve().parents[1] / "config" / "sensor_profiles.yaml"

import perception.terrain_perception.core.algorithms as terrain_perception_algorithms
import planning.global_planner.core.algorithms as global_planner_algorithms
import planning.local_planner.core.algorithms as local_planner_algorithms
import slam.slam.core.algorithms as slam_algorithms

# One (package name in sensor_profiles.yaml's `algorithms` block) -> (that
# package's own algorithms module) -- importing the module registers every
# candidate it declares (see each module's own docstring), so
# `available_algorithms()`/`get_algorithm()` below are asking the real
# source of truth directly, not inferring it from a filesystem convention.
_ALGORITHM_MODULES = {
    "slam": slam_algorithms,
    "terrain_perception": terrain_perception_algorithms,
    "local_planner": local_planner_algorithms,
    "global_planner": global_planner_algorithms,
}


def _load_profiles() -> dict:
    with open(SENSOR_PROFILES_YAML) as f:
        return yaml.safe_load(f)


def test_sensor_profiles_yaml_exists_and_parses():
    profiles = _load_profiles()
    assert isinstance(profiles, dict)
    assert set(profiles) == {"lidar_camera", "camera_only"}


@pytest.mark.parametrize("profile_name", ["lidar_camera", "camera_only"])
def test_profile_has_sensors_and_algorithms_blocks(profile_name):
    profile = _load_profiles()[profile_name]
    assert "sensors" in profile
    assert "algorithms" in profile
    assert set(profile["sensors"]) == {"lidar", "front_camera", "rear_camera", "wrist_camera"}
    assert set(profile["algorithms"]) == set(_ALGORITHM_MODULES)


@pytest.mark.parametrize("profile_name", ["lidar_camera", "camera_only"])
@pytest.mark.parametrize("package_name", list(_ALGORITHM_MODULES))
def test_recommended_algorithm_matches_a_known_candidate_or_is_null(profile_name, package_name):
    # The more likely drift direction: an algorithm gets renamed or removed
    # from its core/algorithms.py but sensor_profiles.yaml still points at
    # the old name.
    algorithm_name = _load_profiles()[profile_name]["algorithms"][package_name]
    if algorithm_name is None:
        return
    known = _ALGORITHM_MODULES[package_name].available_algorithms()
    assert algorithm_name in known, (
        f"sensor_profiles.yaml's {profile_name}.algorithms.{package_name} = "
        f"{algorithm_name!r} has no matching entry in {package_name}/core/algorithms.py "
        f"(known: {known})"
    )


def test_recommended_algorithm_requirements_are_satisfiable_by_its_profiles_sensors():
    # A recommended default that the profile's own sensors can't satisfy
    # would be a silent footgun -- catch it here rather than at launch time.
    # Requirements are read from each algorithm's real registered
    # AlgorithmCfg, not re-typed by hand (which could drift from it).
    profiles = _load_profiles()
    for profile_name, profile in profiles.items():
        for package_name, algorithm_name in profile["algorithms"].items():
            if algorithm_name is None:
                continue
            algorithm_cfg = _ALGORITHM_MODULES[package_name].get_algorithm(algorithm_name)
            for sensor, level in algorithm_cfg.sensor_requirements.items():
                if level == "required":
                    assert profile["sensors"].get(sensor) is True, (
                        f"{profile_name}'s recommended {package_name} algorithm "
                        f"({algorithm_name}) requires {sensor}, which this profile doesn't have"
                    )
