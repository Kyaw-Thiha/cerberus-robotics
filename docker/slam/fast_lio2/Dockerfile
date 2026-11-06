# Cerberus -- SLAM backend image: FAST-LIO2 (LiDAR-only baseline).
#
# ROS2 Jazzy-native -- confirmed 2026-09-21: the vendored fork
# (Kyaw-Thiha/cerberus-spark-fast-lio, a fork of MIT-SPARK/spark-fast-lio)
# is an ament_cmake/colcon package, not ROS1 -- no ros1_bridge needed for
# this backend, unlike fast_livo2/orb_slam3/multicam_slam. Verified building
# locally (package `spark_fast_lio`) after linking Eigen3::Eigen explicitly
# (commit 3b7fdf0 on the fork).
#
# ROS2 is provisioned via pixi (the `slam-fast-lio2` environment in the
# repo's root pixi.toml, robostack-jazzy channel) instead of
# `apt-get install ros-jazzy-*` -- single reproducible dependency source of
# truth shared with local dev. First stage resolves+installs that pixi
# environment from the locked root pixi.toml/pixi.lock; only the resulting
# .pixi/envs/slam-fast-lio2 env and a shell-hook activation script are
# copied into the slim runtime stage below, not the pixi tool itself or the
# full repo. The shared entrypoint (start_slam_backend.sh) sources
# /pixi-project/activate.sh automatically if present.
#
# Source is vendored locally, NOT git-cloned in this Dockerfile -- run
# `python scripts/fetch_slam_backends.py fast_lio2` before building this
# image. This diverges from
# docker/remote/Dockerfile's clone-inside-the-Dockerfile pattern.

FROM ghcr.io/prefix-dev/pixi:noble AS pixi-build

WORKDIR /pixi-project
COPY pixi.toml pixi.lock ./
RUN pixi install -e slam-fast-lio2 --locked

# `pixi shell-hook` emits the activation script the runtime stage sources
# instead of baking `pixi run` into every command below.
RUN pixi shell-hook -e slam-fast-lio2 --shell bash > /pixi-project/activate.sh

FROM ubuntu:noble AS runtime

COPY --from=pixi-build /pixi-project/.pixi/envs/slam-fast-lio2 /pixi-project/.pixi/envs/slam-fast-lio2
COPY --from=pixi-build /pixi-project/activate.sh /pixi-project/activate.sh

RUN mkdir -p /colcon_ws/src
COPY third_party/slam/fast_lio2 /colcon_ws/src/spark_fast_lio

WORKDIR /colcon_ws
RUN /bin/bash -c "source /pixi-project/activate.sh && colcon build --symlink-install --packages-select spark_fast_lio"

COPY docker/slam/shared/start_slam_backend.sh /start_slam_backend.sh
RUN chmod +x /start_slam_backend.sh
ENTRYPOINT ["/start_slam_backend.sh"]
