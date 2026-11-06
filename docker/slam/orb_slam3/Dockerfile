# Cerberus -- SLAM backend image: ORB-SLAM3 (camera-only baseline+latest).
#
# ROS2-native -- the vendored fork (Kyaw-Thiha/cerberus-ORB_SLAM3) ships its
# own core tracking/mapping library (libORB_SLAM3.so, Pangolin-viewer made
# optional, commit 3dd9f33) plus a native ROS2 wrapper package under
# ros2_wrapper/ (ament_cmake, monocular tracking -> nav_msgs/Odometry) --
# no ros1_bridge needed for this backend, unlike fast_livo2/multicam_slam.
# Neither Pangolin nor a display are needed to build/run the core library +
# wrapper (viewer is compiled out entirely when Pangolin isn't present).
#
# ROS2 + OpenCV/cv_bridge + Boost + OpenSSL are provisioned via pixi (the
# `slam-orb-slam3` environment in the repo's root pixi.toml, robostack-jazzy
# + conda-forge channels) instead of `apt-get install ros-jazzy-*` --
# single reproducible dependency source of truth shared with local dev.
# OpenSSL is a real requirement, not carried over out of apt habit: the
# core library's src/System.cc includes <openssl/md5.h> and calls
# MD5_Init/Update/Final directly (settings-file checksum), and
# CMakeLists.txt links -lcrypto -- so the `slam-openssl` feature (conda-forge
# `openssl`) is pulled into this environment specifically for that. First
# stage resolves+installs the pixi environment from the locked root
# pixi.toml/pixi.lock; only the resulting .pixi/envs/slam-orb-slam3 env and
# a shell-hook activation script are copied into the slim runtime stage
# below, not the pixi tool itself or the full repo. The shared entrypoint
# (start_slam_backend.sh) sources /pixi-project/activate.sh automatically
# if present.
#
# Source is vendored locally -- run
# `python scripts/fetch_slam_backends.py orb_slam3` before building.

FROM ghcr.io/prefix-dev/pixi:noble AS pixi-build

WORKDIR /pixi-project
COPY pixi.toml pixi.lock ./
RUN pixi install -e slam-orb-slam3 --locked

# `pixi shell-hook` emits the activation script the runtime stage sources
# instead of baking `pixi run` into every command below.
RUN pixi shell-hook -e slam-orb-slam3 --shell bash > /pixi-project/activate.sh

FROM ubuntu:noble AS runtime

COPY --from=pixi-build /pixi-project/.pixi/envs/slam-orb-slam3 /pixi-project/.pixi/envs/slam-orb-slam3
COPY --from=pixi-build /pixi-project/activate.sh /pixi-project/activate.sh

COPY third_party/slam/orb_slam3 /workspace/orb_slam3

# Build the core library first (its own build.sh, not colcon -- it's a
# plain CMake project, Pangolin optional per commit 3dd9f33), then the
# ROS2 wrapper package that links against it. The pixi env supplies the
# compiler toolchain (cxx-compiler/c-compiler/make/binutils/cmake) that
# build.sh's plain `cmake .. && make` steps run against -- no ROS2 needed
# for this stage, only for the wrapper below.
WORKDIR /workspace/orb_slam3
RUN /bin/bash -c "source /pixi-project/activate.sh && chmod +x build.sh && ./build.sh"

# NOT --symlink-install: confirmed 2026-09-21 (fd04cbe's build log) that a
# symlinked colcon workspace breaks the wrapper's CMakeLists.txt
# CMAKE_SOURCE_DIR/.. resolution back to the core library -- copy the
# package in directly instead of symlinking it into place.
RUN mkdir -p /colcon_ws/src && cp -r /workspace/orb_slam3/ros2_wrapper /colcon_ws/src/orb_slam3_ros2_wrapper

# ros2_wrapper/CMakeLists.txt resolves the core library via
# ${CMAKE_SOURCE_DIR}/../{lib,Thirdparty/DBoW2/lib,Thirdparty/g2o/lib,
# include,include/CameraModels,Thirdparty/Sophus} -- i.e. it expects those
# as SIBLINGS of the wrapper package's own directory. Copying only
# ros2_wrapper/ into /colcon_ws/src breaks that (confirmed by hitting the
# identical failure on multicam_slam's twin-shaped wrapper: "libORB_SLAM3.so
# not found in .../orb_slam3_ros2_wrapper/../lib"). Predates the pixi
# conversion, never exercised by a full end-to-end Docker build until now.
# Symlink the real build output into place instead of copying it.
RUN ln -s /workspace/orb_slam3/lib /colcon_ws/src/lib && \
    ln -s /workspace/orb_slam3/Thirdparty /colcon_ws/src/Thirdparty && \
    ln -s /workspace/orb_slam3/include /colcon_ws/src/include

WORKDIR /colcon_ws
RUN /bin/bash -c "source /pixi-project/activate.sh && colcon build --packages-select orb_slam3_ros2_wrapper"

COPY docker/slam/shared/start_slam_backend.sh /start_slam_backend.sh
RUN chmod +x /start_slam_backend.sh
ENTRYPOINT ["/start_slam_backend.sh"]
