#!/usr/bin/env bash
# Runs ON THE POD, execed by scripts/run_slam_eval_remote.sh from the repo
# root, inside its own subshell that has already sourced
# /pixi-project/activate-gs_livo.sh and /colcon_ws/install/setup.bash.
#
# Unlike fast_lio2's spark_fast_lio (hardcoded "lidar"/"imu" topic names
# needing CLI -r remaps), this fork's LIVMapper reads its subscription
# topics from common.lid_topic/imu_topic/img_topic params -- confirmed by
# reading LIVMapper.cpp's own declare_parameter/get_parameter calls, not
# guessed. Those are already set to the sim's real topics in
# sim_config.yaml, so no remaps are needed here.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# Three real issues found smoke-testing this backend on a pod, 2026-09-25,
# fixed below rather than guessed:
# 1. GS-LIVO shares FAST-LIVO2's package name ("fast_livo") -- this backend
#    MUST run from its own isolated /colcon_ws (not merged with any other
#    backend's), or package files collide. See extract_slam_backend_artifact.sh
#    usage docs for the isolated-workspace sync pattern.
# 2. The synced artifact's install/fast_livo/ is missing its own
#    ament_index resource marker (vikit_common/vikit_ros markers are
#    present, "fast_livo" itself is not -- cause not fully isolated,
#    possibly a docker cp quirk on this specific file during artifact
#    extraction) -- `ros2 pkg list`/`ros2 run` can't find the package
#    without it. Recreated here rather than fixed at extraction time since
#    it's a trivial empty marker file, not worth re-extracting for.
# 3. lib3dgs (a vendored CMake-built dependency under src/gs_livo/src/lib3dgs,
#    never `colcon build`-installed into install/fast_livo/lib/) isn't on
#    the default library search path -- added to LD_LIBRARY_PATH below.
mkdir -p /colcon_ws/install/fast_livo/share/ament_index/resource_index/packages
touch /colcon_ws/install/fast_livo/share/ament_index/resource_index/packages/fast_livo
export AMENT_PREFIX_PATH="/colcon_ws/install/fast_livo:/colcon_ws/install/livox_ros_driver2:${AMENT_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="/colcon_ws/src/gs_livo/src/lib3dgs/build:${LD_LIBRARY_PATH:-}"

# NOT YET FULLY VERIFIED past this point: the pod session that found the 3
# issues above ran out of disk space during this backend's own artifact
# sync (a 13GB pixi env on a tight-disk pod), which silently truncated
# /pixi-project/.pixi/envs/slam-gs-livo/lib/libcudart.so.12 ("file too
# short" at dynamic-load time) -- a corrupted transfer, not a real bug in
# this script or the backend itself. A clean re-sync with adequate disk
# headroom should resolve it; re-verify a clean launch before trusting
# this backend further.
exec ros2 run fast_livo fastlivo_mapping --ros-args \
    --params-file "$REPO_ROOT/slam/slam/backends/gs_livo/sim_config.yaml"
