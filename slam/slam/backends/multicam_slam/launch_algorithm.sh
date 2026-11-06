#!/usr/bin/env bash
# Runs ON THE POD, execed by scripts/run_slam_eval_remote.sh from the repo
# root, inside its own subshell that has already sourced
# /pixi-project/activate-multicam_slam.sh and /colcon_ws/install/setup.bash.
#
# multicam_slam_node's camera1/camera2 rgb topic params default to
# "/camera_01/rgb/image_raw" / "/camera_02/rgb/image_raw" (see
# ros2_wrapper/src/multicam_slam_node.cpp) -- overridden below to point
# directly at this repo's real sim-bridge topic names instead. depth topic
# params are left at their own defaults, since
# stereo_to_depth_launch.py's depth_image_topic params were set to match
# those same defaults (see that file's own comment) -- nothing to override
# here for depth. vocabulary_file points at slam/slam/backends/multicam_slam/
# ORBvoc.txt -- extracted locally (same fix orb_slam3's own vocab needed)
# and synced via the normal repo tar-over-SSH, not the Docker-image-build
# path docker/slam/multicam_slam/Dockerfile also extracts it into (that
# extraction is harmless but redundant now, kept for the case someone runs
# this backend's container standalone rather than layered onto a
# cerberus-remote pod) -- avoids needing a new image build+push before this
# can be smoke-tested.
#
# This node needs stereo_to_depth_launch.py's two stereo_to_depth_node
# instances already running (they produce this node's *_depth_topic
# inputs) -- launched here as background processes, kept alive for this
# script's whole lifetime, rather than folded into
# scripts/run_slam_eval_remote.sh's generic per-backend launch contract
# (every other backend's launch_algorithm.sh is a single exec'd process;
# this is the one backend that genuinely needs two).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# Same RPATH gap confirmed for orb_slam3's identically-shaped wrapper (see
# that backend's own launch_algorithm.sh comment + docker/slam/orb_slam3/
# Dockerfile's "twin-shaped wrapper" note): multicam_slam_ros2_wrapper links
# against libORB_SLAM2.so + Thirdparty/{DBoW2,g2o} via CMake -I/-L paths
# that resolve at build time only, never baked into the binary's own RPATH.
# /workspace/multicam_slam is not part of extract_slam_backend_artifact.sh's
# pulled artifact either (same gap as orb_slam3), so it must be synced
# separately and pointed at explicitly here.
export LD_LIBRARY_PATH="/workspace/multicam_slam/lib:/workspace/multicam_slam/Thirdparty/DBoW2/lib:/workspace/multicam_slam/Thirdparty/g2o/lib:${LD_LIBRARY_PATH:-}"

# NOTE, confirmed on a real pod smoke test 2026-09-25: this launch call
# fails with "package 'slam' not found" against a backend-artifact-only pod
# (one that only has this backend's extracted /colcon_ws, not the main
# repo's own `slam` ROS2 package colcon-built alongside it) -- so the two
# stereo_to_depth_node instances below never actually start, and
# multicam_slam_node's depth topics go unfed in that configuration. Not a
# bug in this script; it's a real prerequisite (the repo's own `slam`
# package must be colcon-built on the target pod, same as any real
# run_slam_eval_remote.sh invocation already does for ground-truth/sensor
# bridge nodes) that a bare artifact-only smoke test doesn't satisfy.
ros2 launch "$REPO_ROOT/slam/slam/backends/multicam_slam/stereo_to_depth_launch.py" &
STEREO_DEPTH_PID=$!
# NOT `exec` for the main node below -- exec replaces this script's own
# process image, which would silently skip this trap (an EXIT trap never
# fires across exec) and leak the two stereo_to_depth_node instances as
# orphans once the SLAM node exits, uncleaned by
# run_slam_eval_remote.sh's own ALGORITHM_PID kill (which only knows about
# whatever PID this whole script ends up as).
trap 'kill "$STEREO_DEPTH_PID" 2>/dev/null' EXIT
sleep 2  # let both stereo_to_depth_node instances finish coming up before the SLAM node starts subscribing to their depth output

ros2 run multicam_slam_ros2_wrapper multicam_slam_node --ros-args \
    -p vocabulary_file:="$REPO_ROOT/slam/slam/backends/multicam_slam/ORBvoc.txt" \
    -p settings_file:="$REPO_ROOT/slam/slam/backends/multicam_slam/sim_config.yaml" \
    -p calibration_file:="$REPO_ROOT/slam/slam/backends/multicam_slam/sim_calibration.txt" \
    -p camera1_rgb_topic:="/front_camera/rgb" \
    -p camera2_rgb_topic:="/rear_camera/rgb"
