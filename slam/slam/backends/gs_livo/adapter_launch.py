"""Thin adapter for GS-LIVO -- see slam/slam/backends/common_adapter.py for
the shared remap/TF/tagging logic. GS-LIVO replaces LVI-Q (2026-09-21, no
public repo existed for LVI-Q) as the LiDAR+camera "latest" candidate --
see slam/slam/core/algorithms.py's gs_livo registration for the full
rationale. A prior map-persistence write-back bug (optimized Gaussian
tensors not being written into the persistent octree map) was found and
fixed upstream; see the "no persistence until that flag is on" framing
below for the current, gated async-export state -- no pod build/run has
verified this end-to-end yet.

Vendored GS-LIVO (github.com/HKUST-Aerial-Robotics/GS-LIVO, forked at
Kyaw-Thiha/cerberus-GS-LIVO) was migrated from ROS1 catkin to a native ROS2
Jazzy ament_cmake package by commits 40c16fd (build system) and af3d2f5
(rclcpp port). The whole backend now runs natively on the ROS2 topic
contract -- it no longer needs ros1_bridge (that
flag was flipped when the migration landed; check
system/cerberus_bringup/config/slam_backends.yaml for current status).

Topic names below are CONFIRMED against the vendored source at
third_party/slam/gs_livo/src/gs-livo/src/LIVMapper.cpp (re-verified against
the rpg_vikit sub-vendoring commit, which shifted these line numbers but did
not change any of the claims below):
- /aft_mapped_to_init -- pubOdomAftMapped advertises nav_msgs::Odometry,
  declared at LIVMapper.cpp:398, called unconditionally from handleLIO()
  (the LIO/LO branch of the main per-cycle dispatch in
  stateEstimationAndMapping()) at :606, published inside
  publish_odometry() at :1610.
- /cloud_registered -- pubLaserCloudFullRes advertises sensor_msgs::PointCloud2,
  declared at LIVMapper.cpp:393. publish_frame_world() is called from TWO
  sites depending on which branch stateEstimationAndMapping() dispatches to
  that cycle: unconditionally from handleVIO() at :520 (the VIO branch, i.e.
  when img_en is true), and gated `if (!img_en)` from handleLIO() at :642
  (the LIO/LO branch). Between the two, it fires every cycle regardless of
  img_en -- publish itself happens inside publish_frame_world() at :1497.
  This is the per-frame registered scan in the world frame, NOT the
  accumulated global map -- GS-LIVO does not publish an accumulated point
  cloud topic (the /Laser_map publisher at :397 is declared but its
  publish() is never called from any code path, and
  /cloud_visual_sub_map_before's publish_visual_sub_map() function at :1547
  is defined but never invoked, so its /cloud_visual_sub_map_before
  publisher at :395 is also dead; /cloud_effected at :396 is published from
  publish_effect_world(), gated behind the pub_effect_point_en param, i.e. a
  debug viz stream). Downstream grid/elevation accumulation is handled
  generically by a shared octomap_server /
  pointcloud_to_grid stage after the contract, not per-backend.

The 3D Gaussian Splat map that distinguishes GS-LIVO from FAST-LIVO2 has no
ROS topic interface at all (purely in-process, lives in vio.cpp's
sub_GSMap + gsmap_manager->gs_map_ octree). Persistence is gated behind
the publish.gs_save_en flag added with the async export work and writes to
Log/GSMap/. See docs/perception.md:143 for
the honest "no persistence until that flag is on" framing.

Frame id: GS-LIVO publishes everything in the "camera_init" frame, which
is its world-fixed odometry origin (NOT a sliding-window frame --
map_sliding_en only prunes the in-memory voxel map, not the world origin).
The common_adapter's blanket rewrite to "map" in common_adapter.py:33 is
correct, not lossy.
"""

import rclpy

from slam.slam.backends.common_adapter import SlamBackendAdapter


def main() -> None:
    rclpy.init()
    node = SlamBackendAdapter(
        node_name="gs_livo_adapter",
        native_odom_topic="/aft_mapped_to_init",  # confirmed: LIVMapper.cpp:398 (decl) + :606 (handleLIO call) + :1610 (publish)
        native_map_topic="/cloud_registered",  # confirmed: LIVMapper.cpp:393 (decl) + :520/:642 (handleVIO/handleLIO calls) + :1497 (publish); per-frame registered scan in world frame -- /Laser_map at :397 and /cloud_visual_sub_map_before at :395 are dead publishers
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
