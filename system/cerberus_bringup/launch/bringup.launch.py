"""Composes the full Cerberus ROS2 graph for a given --platform/--mode.
Resolves platforms.yaml (config/platforms.yaml) into concrete params for
robot_state_publisher and locomotion_policy -- see
docs/superpowers/specs/2026-09-09-multi-platform-config-design.md for the
schema and reasoning. Only this launch file, locomotion_policy, and (once
built) manipulation_policy are platform-aware; every other package gets
robot identity from the standard /robot_description parameter.

Also resolves --sensor_profile (config/sensor_profiles.yaml) and validates
it against each perception/SLAM/planning package's configured algorithm, and
against the chosen locomotion_mode's own sensor_requirements (platforms.yaml)
-- see plans/sensor_setup_handoff.md §4-5. No real slam/terrain_perception/
local_planner/global_planner ROS2 nodes exist yet, so this only validates
and logs the resolved choice; adding the actual Node(...) entries for a
package is a one-line addition to `nodes` below once that package's real
implementation lands, not a restructure.
"""

import importlib
import os
import sys

import yaml
from ament_index_python.packages import get_package_share_directory
from cerberus_bringup.validate_sensor_suite import (
    require_explicit_algorithm_if_unset,
    unimplemented_dependencies,
    validate_sensor_requirements,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node

_SENSOR_PACKAGES = ("slam", "terrain_perception", "local_planner", "global_planner")


def _load_platforms_yaml() -> dict:
    config_path = os.path.join(get_package_share_directory("cerberus_bringup"), "config", "platforms.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_sensor_profiles_yaml() -> dict:
    config_path = os.path.join(get_package_share_directory("cerberus_bringup"), "config", "sensor_profiles.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _launch_setup(context, *args, **kwargs):
    platform_name = LaunchConfiguration("platform").perform(context)
    platforms = _load_platforms_yaml()
    if platform_name not in platforms:
        raise RuntimeError(f"unknown platform {platform_name!r}, known: {sorted(platforms)}")
    platform = platforms[platform_name]

    mode_name = LaunchConfiguration("locomotion_mode").perform(context)
    if not mode_name:
        mode_name = platform["default_locomotion_mode"]
    if mode_name not in platform["locomotion_modes"]:
        raise RuntimeError(
            f"unknown locomotion_mode {mode_name!r} for platform {platform_name!r}, "
            f"known: {sorted(platform['locomotion_modes'])}"
        )
    mode = platform["locomotion_modes"][mode_name]

    description_share = get_package_share_directory("cerberus_description")
    xacro_path = os.path.join(description_share, platform["urdf_relative_path"])

    # CERBERUS_ROOT lets this be overridden explicitly (e.g. non-colcon
    # layouts, containers). When unset, fall back to hopping up from the
    # installed cerberus_bringup package share directory -- this only works
    # for colcon's own install layout, specifically
    # install/cerberus_bringup/share/cerberus_bringup/ -> four ".." hops
    # back to the repo root (install/<pkg>/share/<pkg>/ has depth 4 under
    # the repo root in colcon's --symlink-install and regular install
    # layouts alike).
    repo_root = os.environ.get("CERBERUS_ROOT")
    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(get_package_share_directory("cerberus_bringup"), "..", "..", "..", ".."))
    checkpoint_path = os.path.join(repo_root, mode["checkpoint"])

    joint_names = platform["joint_names"]
    default_joint_pos = [float(v) for v in platform["default_joint_pos"]]

    sensor_profile_name = LaunchConfiguration("sensor_profile").perform(context)
    sensor_profiles = _load_sensor_profiles_yaml()
    if sensor_profile_name not in sensor_profiles:
        raise RuntimeError(f"unknown sensor_profile {sensor_profile_name!r}, known: {sorted(sensor_profiles)}")
    sensor_profile = sensor_profiles[sensor_profile_name]

    # Per-package algorithm: an explicit `--<package>_algorithm` launch arg
    # wins; otherwise fall back to the profile's recommended default (which
    # may itself be `null` -- a documented research gap, see
    # config/sensor_profiles.yaml's comments -- in which case launching that
    # package requires an explicit override).
    resolved_algorithms = {}
    for package_name in _SENSOR_PACKAGES:
        override = LaunchConfiguration(f"{package_name}_algorithm").perform(context)
        chosen = override if override else sensor_profile["algorithms"][package_name]
        resolved_algorithms[package_name] = chosen

    # Registered but not necessarily launched this session -- only validate
    # the packages someone actually asked to run (`--<package>_algorithm` set,
    # or a non-null profile default). This keeps a camera_only run from being
    # forced to explicitly disable slam/local_planner just because their
    # profile default is a documented gap.
    packages_to_launch = {
        pkg: alg
        for pkg, alg in resolved_algorithms.items()
        if LaunchConfiguration(f"{pkg}_algorithm").perform(context) or alg is not None
    }
    for package_name, algorithm_name in packages_to_launch.items():
        require_explicit_algorithm_if_unset(package_name, algorithm_name)

    # Unlike PlatformCfg (kept out of ROS2 launch code because its data --
    # joint limits, collision geometry -- already lives in the URDF, a
    # competing source of truth), an AlgorithmCfg's sensor_requirements has
    # no other home: each module's core/algorithms.py *is* the source of
    # truth. So import it directly rather than re-typing it into
    # sensor_profiles.yaml, which would just invent a second copy that can
    # drift from the real registrations. These packages have no package.xml
    # yet (plain Python, see e.g. slam/slam/__init__.py), so they're never
    # colcon-installed -- repo_root has to be importable regardless of
    # whether this launch file itself is running from source or from an
    # install/ tree.
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    algorithm_getters = {}
    for package_name, dotted_module in (
        ("slam", "slam.slam.core.algorithms"),
        ("terrain_perception", "perception.terrain_perception.core.algorithms"),
        ("local_planner", "planning.local_planner.core.algorithms"),
        ("global_planner", "planning.global_planner.core.algorithms"),
    ):
        # Importing this module registers every candidate it declares --
        # no separate auto-discovery step needed (see that module's own
        # docstring).
        algorithms_module = importlib.import_module(dotted_module)
        algorithm_getters[package_name] = algorithms_module.get_algorithm

    validate_sensor_requirements(
        active_sensors=sensor_profile["sensors"],
        package_requirements={
            "locomotion": mode.get("sensor_requirements", {}),
            **{pkg: algorithm_getters[pkg](alg).sensor_requirements for pkg, alg in packages_to_launch.items()},
        },
    )

    # Non-blocking: a stub dependency (e.g. perceptive mode's height_scan
    # needing a real terrain_perception) is a supported bring-up/testing
    # configuration, not a launch failure -- see requires_implemented's
    # comment in platforms.yaml and unimplemented_dependencies' docstring.
    unimplemented = unimplemented_dependencies(
        mode.get("requires_implemented", []),
        {pkg: algorithm_getters[pkg](alg).implemented for pkg, alg in packages_to_launch.items()},
    )

    nodes = [
        LogInfo(
            msg=(
                f"[cerberus_bringup] sensor_profile={sensor_profile_name} "
                f"algorithms={packages_to_launch}"
            )
        ),
    ]
    if unimplemented:
        nodes.append(
            LogInfo(
                msg=(
                    f"[cerberus_bringup] WARNING: locomotion_mode '{mode_name}' depends on a real "
                    f"implementation of {unimplemented}, which {'is' if len(unimplemented) == 1 else 'are'} "
                    f"still just a declared candidate -- bring-up/testing configuration only, "
                    f"not a validated deployment (see locomotion_policy's own startup warning)."
                )
            )
        )
    nodes += [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": Command(["xacro ", xacro_path])}],
        ),
        Node(
            package="locomotion_policy",
            executable="inference_node",
            name="locomotion_policy",
            output="screen",
            parameters=[
                {"checkpoint_path": checkpoint_path},
                {"observation_shape": mode["observation_shape"]},
                {"joint_names": joint_names},
                {"default_joint_pos": default_joint_pos},
                {"num_joints": len(joint_names)},
                {"height_scan_size": mode["height_scan_size"]},
                {"action_scale": mode["action_scale"]},
            ],
        ),
    ]
    return nodes


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("platform", default_value="go2", description="Robot platform (see config/platforms.yaml)."),
        DeclareLaunchArgument(
            "locomotion_mode",
            default_value="",
            description="blind or perceptive -- defaults to the platform's own default_locomotion_mode if empty.",
        ),
        DeclareLaunchArgument(
            "sensor_profile",
            default_value="lidar_camera",
            description="Sensor rig (see config/sensor_profiles.yaml): lidar_camera or camera_only.",
        ),
    ]
    for package_name in _SENSOR_PACKAGES:
        args.append(
            DeclareLaunchArgument(
                f"{package_name}_algorithm",
                default_value="",
                description=(
                    f"Algorithm override for {package_name} (see that package's "
                    f"core/algorithms.py) -- defaults to the active sensor_profile's "
                    f"recommended algorithm if empty."
                ),
            )
        )
    args.append(OpaqueFunction(function=_launch_setup))
    return LaunchDescription(args)
