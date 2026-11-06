#!/usr/bin/env bash
# Shared entrypoint for every docker/slam/<name>/Dockerfile image -- sources
# the built workspace and execs whatever launch command is passed in, so
# each image stays a plain `docker run <image> roslaunch ...` / `ros2 launch
# ...` invocation rather than needing a bespoke entrypoint per backend.
set -euo pipefail

# pixi-provisioned images (fast_lio2, bievr_lio, orb_slam3, multicam_slam,
# fast_livo2, 2026-09-21) copy in a pixi shell-hook activation script instead
# of relying on an apt-installed /opt/ros/jazzy -- source it FIRST, since the
# colcon workspace's own install/setup.bash assumes the pixi/conda env's
# Python + libs (rclpy, cv_bridge, etc.) are already on PATH/LD_LIBRARY_PATH.
# gs_livo stays apt/source-build (still ROS1, CUDA/LibTorch-dependent) and
# has no /pixi-project/activate.sh, so this is a no-op there.
if [[ -f /pixi-project/activate.sh ]]; then
    # One of pixi's own conda activation hooks (ros-jazzy-ros-workspace's
    # activate.d script) references $CONDA_BUILD with no default -- fine
    # under plain bash, fatal here under `set -u` ("CONDA_BUILD: unbound
    # variable"). Confirmed by hitting this exact failure running a
    # pixi-provisioned image through this entrypoint (2026-09-21). Relax
    # -u only around the source itself, not the rest of this script.
    set +u
    # shellcheck disable=SC1091
    source /pixi-project/activate.sh
    set -u
fi

# colcon's own generated install/setup.bash has the identical class of
# problem: it references $COLCON_TRACE with no default ("COLCON_TRACE:
# unbound variable" under `set -u`), same as pixi's activation hook above --
# confirmed by hitting this exact failure too. Relax -u around both sourced
# scripts, not just pixi's.
set +u
if [[ -f /catkin_ws/devel/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /catkin_ws/devel/setup.bash
elif [[ -f /colcon_ws/install/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /colcon_ws/install/setup.bash
fi
set -u

exec "$@"
