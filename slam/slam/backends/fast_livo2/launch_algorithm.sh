#!/usr/bin/env bash
# Runs ON THE POD, execed by scripts/run_slam_eval_remote.sh from the repo
# root, inside its own subshell that has already sourced
# /pixi-project/activate-fast_livo2.sh and /colcon_ws/install/setup.bash.
#
# FAST-LIVO2's real launch topology (confirmed by reading
# third_party/slam/fast_livo2/launch/mapping_avia_marslvig.launch.py
# directly, not guessed) is TWO nodes, not one: a `parameter_blackboard`
# node holding the camera intrinsics (fastlivo_mapping reads them via a
# cross-node parameter client, not its own --params-file), plus the main
# `fastlivo_mapping` node itself with the lidar/imu/vio config. lid_topic/
# imu_topic/img_topic ARE parameterized here (unlike fast_lio2's lidar/imu,
# which needed CLI -r remaps) -- confirmed via LIVMapper.cpp's own
# declare_parameter("common.lid_topic"/"imu_topic"/"img_topic") calls, so
# sim_config.yaml sets them directly and no -r remap is needed.
#
# The vendored launch file also runs an image_transport `republish`
# compressed->raw node -- not needed here since our sim already publishes
# raw sensor_msgs/Image directly on /front_camera/rgb, so img_topic points
# straight at that.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# ROOT CAUSE (confirmed 2026-09-26 on a real pod, not guessed): the previous
# "Camera model not correctly specified" failure was NEVER a discovery-timing
# problem, even though it looked like one (hardcoded 100ms wait_for_service,
# a natural first suspect). Proof: instrumenting the real fastlivo_mapping
# binary with `--ros-args --log-level debug` showed vikit_ros's
# rclcpp::SyncParametersClient constructing SIX separate DDS client
# services simultaneously (get_parameters, get_parameter_types,
# set_parameters, set_parameters_atomically, list_parameters,
# describe_parameters -- SyncParametersClient always creates all six, even
# though only get_parameters is ever called), and ALL SIX consistently
# timed out at EXACTLY the 100ms mark -- even after giving
# parameter_blackboard.py a 30-second head start (far beyond the 5s here).
# A minimal rclpy reproduction doing the equivalent single-service
# get_parameters call (see the throwaway diagnostic used to isolate this)
# succeeded in well under 100ms even with ZERO lead time, proving the
# blackboard side and general DDS discovery are both fine -- the failure is
# specific to Fast-DDS's default SIMPLE discovery protocol needing to
# announce/match six brand-new endpoint pairs at once (created fresh the
# instant SyncParametersClient is constructed, regardless of how long the
# process has been alive), which apparently doesn't reliably complete
# within 100ms. Switching the RMW implementation to Cyclone DDS (already
# present in this backend's pixi env, no rebuild needed) fixed it
# reproducibly across repeated trials: fastlivo_mapping loads the correct
# camera intrinsics (Pinhole, 733/733/320/240) and comes up fully
# (/laserMapping node subscribing to /lidar/points, /imu/data,
# /front_camera/rgb and publishing /aft_mapped_to_init, /Laser_map, /path,
# etc.) every time. Applies to BOTH processes below so they speak the same
# RMW/wire protocol.
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# `ros2 run demo_nodes_cpp parameter_blackboard` -- demo_nodes_cpp is not a
# dependency of this backend's pixi env (confirmed: "Package 'demo_nodes_cpp'
# not found" against the real synced artifact) and isn't worth a Docker
# rebuild for. parameter_blackboard.py is a faithful minimal reimplementation
# -- see its own docstring.
python3 "$REPO_ROOT/slam/slam/backends/fast_livo2/parameter_blackboard.py" --ros-args \
    --params-file "$REPO_ROOT/slam/slam/backends/fast_livo2/camera_config.yaml" &
BLACKBOARD_PID=$!
# No EXIT trap here -- exec below replaces this shell's process image, which
# drops any trap table with it. The blackboard node is left running after
# this script's own process ends (orphaned, reparented) -- harmless (no
# lasting state, whole pod gets torn down after eval anyway), just not
# auto-cleaned by run_slam_eval_remote.sh's own ALGORITHM_PID kill, which
# only ever sees this script's own PID (pre-exec) or fastlivo_mapping's
# (post-exec).

sleep 5  # blackboard startup lead time -- kept even though the actual bug (above) turned out to be
# RMW-specific rather than timing-specific; harmless margin, cheap insurance against any genuine
# first-connection latency.

exec ros2 run fast_livo fastlivo_mapping --ros-args \
    --params-file "$REPO_ROOT/slam/slam/backends/fast_livo2/sim_config.yaml"
