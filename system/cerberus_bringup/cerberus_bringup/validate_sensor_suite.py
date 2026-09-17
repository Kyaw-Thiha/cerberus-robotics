"""Fail-loud sensor-suite/algorithm compatibility check. Extends
plans/sensor_setup_handoff.md §4's `validate()` sketch: instead of a
hardcoded package->sensor manifest, requirements come from whichever
`AlgorithmCfg` each package is configured to run (see
config/sensor_profiles.yaml's `algorithms` block and each module's own
`core/algorithms.py`). Pure function, no ROS2 dependency -- called
by bringup.launch.py's `_launch_setup` before spawning any node, and
unit-tested directly here without needing `ros2 launch`.
"""

from __future__ import annotations


class SensorSuiteMismatchError(Exception):
    pass


def validate_sensor_requirements(
    active_sensors: dict[str, bool],
    package_requirements: dict[str, dict[str, str]],
) -> None:
    """`active_sensors`: the resolved profile's `sensors` block (from
    sensor_profiles.yaml), e.g. {"lidar": True, "front_camera": True, ...}.
    `package_requirements`: {package_name: AlgorithmCfg.sensor_requirements}
    for every package about to be launched. Raises `SensorSuiteMismatchError`
    naming exactly which package needed exactly which missing sensor --
    turns a silently-hanging subscriber into a startup error, per
    sensor_setup_handoff.md §4's stated goal.
    """
    missing = []
    for package, requirements in package_requirements.items():
        for sensor, level in requirements.items():
            if level == "required" and not active_sensors.get(sensor, False):
                missing.append((package, sensor))
    if missing:
        detail = ", ".join(f"{package} needs {sensor}" for package, sensor in missing)
        raise SensorSuiteMismatchError(
            f"Active sensor profile is missing sensors required by: {detail}. "
            f"Either switch --sensor_profile or don't launch these packages."
        )


def unimplemented_dependencies(required_packages: list[str], implemented_flags: dict[str, bool]) -> list[str]:
    """Which of `required_packages` currently resolve to a stub (declared
    AlgorithmCfg, no real code -- `implemented=False`) rather than a real
    implementation. Non-blocking on purpose, unlike validate_sensor_requirements:
    a missing sensor makes a launch pointless, but a stub dependency (e.g.
    locomotion's `perceptive` mode depending on a real terrain_perception)
    is an already-supported bring-up/testing configuration -- see
    inference_node.py's height_scan stub. Callers should warn, not raise.
    `implemented_flags`: {package_name: AlgorithmCfg.implemented} for the
    algorithm each `required_packages` entry actually resolved to; a
    package missing from this dict (e.g. not launched at all) counts as
    unimplemented too.
    """
    return [pkg for pkg in required_packages if not implemented_flags.get(pkg, False)]


def require_explicit_algorithm_if_unset(package: str, algorithm: str | None) -> str:
    """Raises if a profile has no recommended algorithm for `package` (a
    `null` entry in sensor_profiles.yaml's `algorithms` block -- a real,
    flagged research gap per docs/perception.md, not a bug) and the caller
    didn't pass an explicit override. Keeps "no candidate documented yet"
    from silently resolving to launching nothing or crashing on a `None`
    checkpoint/module path deeper in the graph.
    """
    if algorithm is None:
        raise SensorSuiteMismatchError(
            f"No recommended algorithm is configured for '{package}' under this sensor profile "
            f"(see config/sensor_profiles.yaml) -- pass --{package}_algorithm explicitly, "
            f"or don't launch this package."
        )
    return algorithm
