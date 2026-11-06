"""Wires multicam_slam's real stereo-to-depth pipeline into the ROS2 graph:
the front and rear stereo camera pairs -> two `stereo_to_depth_node`
instances (see that module's docstring for why one instance per camera
position, and why this is a separate node rather than backend-specific logic
folded into `common_adapter.SlamBackendAdapter`) -> the front/rear rgb+depth
topics multicam_slam's ROS2 wrapper subscribes to. Launched ALONGSIDE
`adapter_launch.py`'s `multicam_slam_adapter` (pose/map remap only), not by
it -- this graph produces the algorithm's INPUT topics, the adapter consumes
its OUTPUT topics; the two don't depend on each other's node lifecycle.

A plain `generate_launch_description()` module, composed into
system/cerberus_bringup's bringup.launch.py rather than installed for
standalone `ros2 launch slam ...` invocation -- this package's setup.py has no
`launch/` data_files entry.

`left_image_topic`/`right_image_topic` point directly at this repo's own
sim-bridge topic names (/front_camera/rgb, /front_camera_right/rgb, and
the rear equivalents -- see slam/slam/ground_truth/sim_ipc.py's
TOPIC_CAMERA_FRONT_RIGHT_RGB/TOPIC_CAMERA_REAR_RIGHT_RGB), not the original
ROS1-wrapper-derived "/camera_01/..." placeholder names -- confirmed against
the real vendored ros2_wrapper/src/multicam_slam_node.cpp source (its
camera1_rgb_topic/camera2_rgb_topic params are overridden to match these
same names in slam/slam/backends/multicam_slam/launch_algorithm.sh, so
neither node needs to agree on the old placeholder naming at all).
`depth_image_topic` is kept at multicam_slam_node.cpp's own
camera1_depth_topic/camera2_depth_topic PARAM DEFAULTS
("/camera_01/depth/image_raw" / "/camera_02/depth/image_raw") specifically
so launch_algorithm.sh doesn't need to override those two params --
`right_image_topic` stays purely internal either way (the SLAM node never
subscribes to it).
"""

from __future__ import annotations

from launch import LaunchDescription
from launch_ros.actions import Node

# baseline_m/focal_length_mm/horizontal_aperture_mm/image_width_px are left
# at stereo_to_depth_node.py's own defaults (sourced from
# common/cerberus_description/sensors/sensor_configs.py) below -- only the
# per-position topic names differ between the two instances.
_FRONT_PARAMS = {
    "left_image_topic": "/front_camera/rgb",
    "right_image_topic": "/front_camera_right/rgb",
    "depth_image_topic": "/camera_01/depth/image_raw",
}
_REAR_PARAMS = {
    "left_image_topic": "/rear_camera/rgb",
    "right_image_topic": "/rear_camera_right/rgb",
    "depth_image_topic": "/camera_02/depth/image_raw",
}


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="slam",
                executable="stereo_to_depth_node",
                name="stereo_to_depth_front",
                output="screen",
                parameters=[_FRONT_PARAMS],
            ),
            Node(
                package="slam",
                executable="stereo_to_depth_node",
                name="stereo_to_depth_rear",
                output="screen",
                parameters=[_REAR_PARAMS],
            ),
        ]
    )
