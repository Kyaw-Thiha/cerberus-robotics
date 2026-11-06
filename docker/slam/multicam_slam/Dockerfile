# Cerberus -- SLAM backend image: Multicam-SLAM (camera-only, front+rear
# dual-RGBD fusion, latest -- fuses on top of ORB-SLAM2, per
# the corrected sensor description: this is dual-RGBD, NOT
# front+rear stereo as originally assumed. The RGBD input is fed by
# stereo-computed depth from slam/slam/backends/multicam_slam/
# stereo_to_depth_node.py in the main repo, not a real RGB-D sensor --
# see that module's docstring and docs/perception.md's stereo-over-RGB-D
# decision for why.
#
# ROS2 Jazzy-native -- confirmed 2026-09-21 (fork commits 2cd67e3, 2583817
# on Kyaw-Thiha/cerberus-Multi_ORB_SLAM): the vendored fork's core algorithm
# library builds against modern CMake/OpenCV4/Eigen3 with Pangolin made
# optional, and ships a native ROS2 ros2_wrapper/ package (ament_cmake,
# dual-RGBD tracking -> nav_msgs/Odometry, publishing live pose the
# original ROS1 ros_rgbd.cc never did) -- no ros1_bridge needed, unlike
# gs_livo (still ROS1, migration not yet started).
#
# ROS2 + OpenCV/cv_bridge + Boost are provisioned via pixi (the
# `slam-multicam-slam` environment in the repo's root pixi.toml,
# robostack-jazzy + conda-forge channels) instead of
# `apt-get install ros-jazzy-*` -- single reproducible dependency source of
# truth shared with local dev. Unlike orb_slam3, this fork has no OpenSSL
# dependency (confirmed by grep -rniE "openssl|libssl|libcrypto" over the
# vendored tree -- no hits), so `slam-multicam-slam` stays on the same
# slam-base + slam-cv + slam-boost features as before, no extra OpenSSL
# feature needed. First stage resolves+installs the pixi environment from
# the locked root pixi.toml/pixi.lock; only the resulting
# .pixi/envs/slam-multicam-slam env and a shell-hook activation script are
# copied into the slim runtime stage below, not the pixi tool itself or the
# full repo. The shared entrypoint (start_slam_backend.sh) sources
# /pixi-project/activate.sh automatically if present.
#
# Source is vendored locally -- run
# `python scripts/fetch_slam_backends.py multicam_slam` before building.

FROM ghcr.io/prefix-dev/pixi:noble AS pixi-build

WORKDIR /pixi-project
COPY pixi.toml pixi.lock ./
RUN pixi install -e slam-multicam-slam --locked

# `pixi shell-hook` emits the activation script the runtime stage sources
# instead of baking `pixi run` into every command below.
RUN pixi shell-hook -e slam-multicam-slam --shell bash > /pixi-project/activate.sh

FROM ubuntu:noble AS runtime

COPY --from=pixi-build /pixi-project/.pixi/envs/slam-multicam-slam /pixi-project/.pixi/envs/slam-multicam-slam
COPY --from=pixi-build /pixi-project/activate.sh /pixi-project/activate.sh

COPY third_party/slam/multicam_slam /workspace/multicam_slam

# Build the core library and its vendored Thirdparty/{DBoW2,g2o} first (its
# own build.sh, not colcon -- plain CMake, Pangolin optional per commit
# 2cd67e3), then the ROS2 wrapper package that links against it. The pixi
# env supplies the compiler toolchain (cxx-compiler/c-compiler/make/
# binutils/cmake) that build.sh's plain `cmake .. && make` steps run
# against -- no ROS2 needed for this stage, only for the wrapper below.
WORKDIR /workspace/multicam_slam
RUN /bin/bash -c "source /pixi-project/activate.sh && bash build.sh"

# NOT --symlink-install: same CMAKE_SOURCE_DIR/.. resolution concern
# confirmed for orb_slam3's identically-shaped wrapper -- copy the package
# in directly instead of symlinking it into place.
RUN mkdir -p /colcon_ws/src && cp -r /workspace/multicam_slam/ros2_wrapper /colcon_ws/src/multicam_slam_ros2_wrapper

# ros2_wrapper/CMakeLists.txt resolves the core library via
# ${CMAKE_SOURCE_DIR}/../{lib,Thirdparty/DBoW2/lib,Thirdparty/g2o/lib,include}
# -- i.e. it expects lib/, Thirdparty/, include/ as SIBLINGS of the wrapper
# package's own directory. Copying only ros2_wrapper/ into /colcon_ws/src
# breaks that: "../.." from /colcon_ws/src/multicam_slam_ros2_wrapper is
# /colcon_ws/src, which never had those directories -- confirmed by hitting
# exactly this ("libORB_SLAM2.so not found in
# /colcon_ws/src/multicam_slam_ros2_wrapper/../lib") on a real build. This
# predates the pixi conversion (same cp -r structure in the original
# apt-based Dockerfile) -- it was never exercised by an actual end-to-end
# Docker build until now. Symlink the real build output into place instead
# of copying it (cheap, and colcon safely ignores non-package directories
# during its own package discovery).
RUN ln -s /workspace/multicam_slam/lib /colcon_ws/src/lib && \
    ln -s /workspace/multicam_slam/Thirdparty /colcon_ws/src/Thirdparty && \
    ln -s /workspace/multicam_slam/include /colcon_ws/src/include

WORKDIR /colcon_ws
RUN /bin/bash -c "source /pixi-project/activate.sh && colcon build --packages-select multicam_slam_ros2_wrapper"

# The ORB vocabulary (third_party/slam/multicam_slam/Vocabulary/ORBvoc.txt.tar.gz,
# already on disk from the COPY above) never reaches a pod through any other
# mechanism: third_party/ is gitignored, vendored-source-only, and never synced
# to a running pod (only the extracted colcon_ws/pixi-env artifact is, via
# scripts/extract_slam_backend_artifact.sh's `docker cp CONTAINER:/colcon_ws`).
# Extracting it into /colcon_ws here makes it ride along with that same
# extraction for free -- confirmed as a real, previously-missing gap by tracing
# how the artifact actually reaches a pod, not assumed.
RUN mkdir -p /colcon_ws/vocabulary && \
    tar xzf /workspace/multicam_slam/Vocabulary/ORBvoc.txt.tar.gz -C /colcon_ws/vocabulary

COPY docker/slam/shared/start_slam_backend.sh /start_slam_backend.sh
RUN chmod +x /start_slam_backend.sh
ENTRYPOINT ["/start_slam_backend.sh"]
