#!/usr/bin/env bash
# Runs ON THE POD, execed by scripts/run_slam_eval_remote.sh from the repo
# root, inside its own subshell that has already sourced
# /pixi-project/activate-orb_slam3.sh and /colcon_ws/install/setup.bash.
#
# orb_slam3_node.cpp's `image_raw` subscription is a hardcoded relative
# topic name (confirmed by reading ros2_wrapper/src/orb_slam3_node.cpp
# directly), so the sim's real image topic is wired with a `-r` CLI remap
# here, same reason fast_lio2/launch_algorithm.sh remaps `lidar`/`imu`.
#
# vocabulary_file points at ORBvoc.txt extracted locally from
# third_party/slam/orb_slam3/Vocabulary/ORBvoc.txt.tar.gz (138MB
# uncompressed) into this same directory and gitignored -- it is NOT part
# of the extracted Docker artifact (extract_slam_backend_artifact.sh only
# pulls /colcon_ws + the pixi env, not /workspace/orb_slam3, where the
# Dockerfile's own COPY of third_party/slam/orb_slam3 puts the still-tarred
# copy) -- so it has to reach the pod through this repo's own normal
# tar-over-SSH code sync instead, same path sim_settings.yaml already
# travels.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# orb_slam3_node links against libORB_SLAM3.so via a relative `-l` at build
# time, not an absolute path or an RPATH entry pointing at it (confirmed via
# `readelf -d`: RPATH is only "/pixi-project/.pixi/envs/slam-orb-slam3/lib",
# nothing under /workspace) -- so even though the Dockerfile's own symlinks
# (`/colcon_ws/src/lib` -> `/workspace/orb_slam3/lib`, same for
# Thirdparty/include) make the .so resolvable at BUILD time via
# CMake's own -I/-L search paths, nothing carries that path forward into the
# built binary for RUNTIME resolution. /workspace/orb_slam3/lib is NOT part
# of extract_slam_backend_artifact.sh's pulled artifact (that script only
# grabs /colcon_ws + the pixi env -- see this script's own vocabulary_file
# comment above for the identical reason), so it must be synced separately
# (see scripts/extract_slam_backend_artifact.sh's usage docs) and pointed at
# explicitly here. Same fix needed for multicam_slam's twin-shaped wrapper,
# per docker/slam/orb_slam3/Dockerfile's own comment.
# Same story for its two vendored Thirdparty deps (DBoW2, g2o) -- confirmed
# by re-running after the fix above and hitting the identical error one
# level down, for libDBoW2.so, then libg2o.so.
export LD_LIBRARY_PATH="/workspace/orb_slam3/lib:/workspace/orb_slam3/Thirdparty/DBoW2/lib:/workspace/orb_slam3/Thirdparty/g2o/lib:${LD_LIBRARY_PATH:-}"

exec ros2 run orb_slam3_ros2_wrapper orb_slam3_node --ros-args \
    -p vocabulary_file:="$REPO_ROOT/slam/slam/backends/orb_slam3/ORBvoc.txt" \
    -p settings_file:="$REPO_ROOT/slam/slam/backends/orb_slam3/sim_settings.yaml" \
    -r image_raw:=/front_camera/rgb
