# Cerberus -- SLAM backend image: BIEVR-LIO (LiDAR-only latest).
#
# ROS2 Jazzy-native -- confirmed against its own README:
# ships bievr_lio_ros2, no ros1_bridge needed.
#
# ROS2 + Ceres are provisioned via pixi (the `slam-bievr-lio` environment in
# the repo's root pixi.toml, robostack-jazzy + conda-forge channels) instead
# of `apt-get install ros-jazzy-*` -- single reproducible dependency source
# of truth shared with local dev. Ceres 2.2.0 comes from conda-forge's
# `ceres-solver` package (exact version match), which replaces this
# backend's from-source Ceres build (docker/scripts/install_ceres.sh in the
# vendored source) outright -- no source build needed here anymore. First
# stage resolves+installs the pixi environment from the locked root
# pixi.toml/pixi.lock; only the resulting .pixi/envs/slam-bievr-lio env and
# a shell-hook activation script are copied into the slim runtime stage
# below. The shared entrypoint (start_slam_backend.sh) sources
# /pixi-project/activate.sh automatically if present.
#
# Vendored from the Cerberus fork (Kyaw-Thiha/cerberus-BIEVR-LIO, forked
# from ethz-asl/BIEVR-LIO), which carries a real, permanent fix (commit
# 9a6ec0b): BOTH BIEVR/CMakeLists.txt (the bievr_lio package) AND
# interfaces/ros_common/CMakeLists.txt (the bievr_ros_common package it
# depends on) originally declared `LANGUAGES CXX` without C -- C was never
# enabled as a build language in either. conda-forge's ceres-solver is built
# with real SuiteSparse support, and SuiteSparse's exported CMake config
# transitively calls `find_package(OpenMP COMPONENTS C)`, which fails
# unless the consuming project has called enable_language(C) -- confirmed
# by reproducing this standalone (a scratch CMakeLists.txt without
# `LANGUAGES CXX C` hits the identical "Could NOT find OpenMP (missing:
# OpenMP_C_FOUND C)" error, and adding C fixes it) AND by hitting it twice
# on a real build: first in bievr_lio itself, then again in
# bievr_ros_common once bievr_lio's own fix let the build get further and
# transitively pull in bievr_lioConfig.cmake -> Ceres -> SuiteSparse the
# same way. The old from-source Ceres build never hit either because the
# apt image never installed libsuitesparse-dev, so Ceres silently built
# without real SuiteSparse support -- conda-forge's package is simply more
# complete, which is what exposes the gap.
#
# Source is vendored locally -- run
# `python scripts/fetch_slam_backends.py bievr_lio` before building.

FROM ghcr.io/prefix-dev/pixi:noble AS pixi-build

WORKDIR /pixi-project
COPY pixi.toml pixi.lock ./
RUN pixi install -e slam-bievr-lio --locked

# `pixi shell-hook` emits the activation script the runtime stage sources
# instead of baking `pixi run` into every command below.
RUN pixi shell-hook -e slam-bievr-lio --shell bash > /pixi-project/activate.sh

FROM ubuntu:noble AS runtime

COPY --from=pixi-build /pixi-project/.pixi/envs/slam-bievr-lio /pixi-project/.pixi/envs/slam-bievr-lio
COPY --from=pixi-build /pixi-project/activate.sh /pixi-project/activate.sh

COPY third_party/slam/bievr_lio /colcon_ws/src/BIEVR-LIO

WORKDIR /colcon_ws
RUN /bin/bash -c "source /pixi-project/activate.sh && colcon build --packages-up-to bievr_lio_ros2"

COPY docker/slam/shared/start_slam_backend.sh /start_slam_backend.sh
RUN chmod +x /start_slam_backend.sh
ENTRYPOINT ["/start_slam_backend.sh"]
