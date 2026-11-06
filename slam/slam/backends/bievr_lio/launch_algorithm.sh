#!/usr/bin/env bash
# Runs ON THE POD, execed by scripts/run_slam_eval_remote.sh from the repo
# root, inside its own subshell that has already sourced
# /pixi-project/activate-bievr_lio.sh and /colcon_ws/install/setup.bash.
#
# Unlike fast_lio2's spark_lio_mapping (hardcoded relative "lidar"/"imu"
# topic names needing CLI -r remaps), bievr_lio_ros2's process_topics reads
# its subscription topics from the sensor-config YAML's own `topics:`
# section (process_topics.cpp: config.topic_config.pointcloud_topic /
# imu_topic) -- so this launch needs no -r remaps, just the right config
# file passed through. Launched via `ros2 launch`, not `ros2 run`, because
# process_topics.launch.py's resolve_config() is what turns our config path
# into the --sensor_config_file/--params_file CLI arguments process_topics
# actually parses (plain yaml-cpp, not ROS2's --ros-args mechanism -- see
# process_topics.cpp).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
exec ros2 launch bievr_lio_ros2 process_topics.launch.py \
    sensor_config:="$REPO_ROOT/slam/slam/backends/bievr_lio/sim_config.yaml" \
    params:=params \
    rviz:=false
