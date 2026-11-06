#!/usr/bin/env bash
# Runs ON THE POD, execed by scripts/run_slam_eval_remote.sh from the repo
# root, inside its own subshell that has already sourced
# /pixi-project/activate-fast_lio2.sh and /colcon_ws/install/setup.bash.
#
# spark_lio_mapping's `lidar`/`imu` subscriptions are hardcoded relative
# topic names in spark_fast_lio.cpp (NOT parameterized via a `common.*_topic`
# param, despite that param existing in some of the vendored example configs
# -- confirmed by reading create_subscription<...>("lidar", ...) /
# ("imu", ...) directly), so the sim's real topics are wired with `-r`
# CLI remaps here rather than in sim_config.yaml.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
exec ros2 run spark_fast_lio spark_lio_mapping --ros-args \
    --params-file "$REPO_ROOT/slam/slam/backends/fast_lio2/sim_config.yaml" \
    -r lidar:=/lidar/points -r imu:=/imu/data
