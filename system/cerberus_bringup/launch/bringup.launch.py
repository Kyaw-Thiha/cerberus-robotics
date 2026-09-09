"""Composes the full Cerberus ROS2 graph for a given --platform/--mode.
Resolves platforms.yaml (config/platforms.yaml) into concrete params for
robot_state_publisher and locomotion_policy -- see
docs/superpowers/specs/2026-09-09-multi-platform-config-design.md for the
schema and reasoning. Only this launch file, locomotion_policy, and (once
built) manipulation_policy are platform-aware; every other package gets
robot identity from the standard /robot_description parameter.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def _load_platforms_yaml() -> dict:
    config_path = os.path.join(get_package_share_directory("cerberus_bringup"), "config", "platforms.yaml")
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

    nodes = [
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
    return LaunchDescription(
        [
            DeclareLaunchArgument("platform", default_value="go2", description="Robot platform (see config/platforms.yaml)."),
            DeclareLaunchArgument(
                "locomotion_mode",
                default_value="",
                description="blind or perceptive -- defaults to the platform's own default_locomotion_mode if empty.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
