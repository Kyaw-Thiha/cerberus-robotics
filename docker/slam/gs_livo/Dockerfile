# Cerberus -- SLAM backend image: GS-LIVO (LiDAR+camera latest, replaces
# LVI-Q as of 2026-09-21).
#
# ROS2 Jazzy -- the vendored fork (Kyaw-Thiha/cerberus-GS-LIVO) was migrated
# from ROS1 catkin to native ament_cmake/rclcpp (fork commits 40c16fd,
# af3d2f5); this Dockerfile was written/verified AFTER that migration, not
# before it -- earlier versions of this file describing a still-ROS1 image
# were stale and have been replaced.
#
# GENUINELY GPU-DEPENDENT AT BUILD TIME, unlike every other SLAM backend
# here: the real-time 3D-Gaussian-Splatting mapping component (src/lib3dgs/)
# has CUDA kernels that must be compiled with nvcc, and links against
# LibTorch. This image can only be built on a machine (or RunPod pod) with
# a CUDA-capable GPU and toolkit.
#
# ROS2 + CUDA toolkit + LibTorch are ALL provisioned via pixi (the
# `slam-gs-livo` environment in the repo's root pixi.toml, robostack-jazzy +
# conda-forge channels) -- same single-dependency-source-of-truth pattern
# as the other 5 backends, extended with conda-forge's cuda-toolkit/
# cuda-nvcc/libtorch packages instead of a manually-downloaded LibTorch zip
# or an nvidia/cuda base image. Two things this needed beyond the pattern
# established by fast_livo2 (the next-closest backend in complexity):
#   - A pinned gcc/g++ 12 toolchain (the `slam-gs-livo-extra` feature) --
#     CUDA 12.x's nvcc rejects host compilers newer than GCC 12, and the
#     shared slam-base feature's cxx-compiler/c-compiler resolve to GCC 15
#     by default.
#   - cuda-toolkit pinned to 12.9.* to match the resolved libtorch build's
#     own cuda129 tag -- an unpinned cuda-toolkit could otherwise resolve a
#     newer CUDA runtime than the libtorch build was actually compiled
#     against.
#
# CONFIRMED building clean end-to-end (2026-09-22, fork commits f52146e,
# b5ec0a3): actually built on a real RunPod RTX 4090 pod (not just
# reasoned about) -- tinyply, Livox-SDK2, livox_ros_driver2, lib3dgs's CUDA
# Gaussian-rasterizer kernels, and fastlivo_mapping itself all compile and
# link clean, producing a real install/fast_livo/lib/fast_livo/
# fastlivo_mapping executable. See b5ec0a3's commit message for the full
# list of ROS2/CMake/LibTorch API fixes this took (none of it had ever
# been build-tested before that commit). GS-map persistence itself (see
# slam/slam/core/algorithms.py's gs_livo registration and slam/slam/
# backends/gs_livo/adapter_launch.py for the full detail) is a gated,
# off-by-default async .ply export -- the write-back bug is fixed
# (commits 34ec00e, 4a76dad) and the export path itself now builds, but
# no pod RUN (only build) has exercised it against live sensor data yet.
#
# Source is vendored locally -- run
# `python scripts/fetch_slam_backends.py gs_livo` before building.

FROM ghcr.io/prefix-dev/pixi:noble AS pixi-build

WORKDIR /pixi-project
COPY pixi.toml pixi.lock ./
RUN pixi install -e slam-gs-livo --locked

# `pixi shell-hook` emits the activation script the runtime stage sources
# instead of baking `pixi run` into every command below.
RUN pixi shell-hook -e slam-gs-livo --shell bash > /pixi-project/activate.sh

FROM ubuntu:noble AS runtime

ENV DEBIAN_FRONTEND=noninteractive
# ca-certificates is required alongside git: ubuntu:noble ships no CA store
# by default, so `git clone https://...` below fails with "server
# certificate verification failed" without it -- same fix as fast_livo2's
# Dockerfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=pixi-build /pixi-project/.pixi/envs/slam-gs-livo /pixi-project/.pixi/envs/slam-gs-livo
COPY --from=pixi-build /pixi-project/activate.sh /pixi-project/activate.sh

# The pixi env's cmake is much newer than any Livox-SDK2/livox_ros_driver2
# release ever targeted, and CMake >=4.0 hard-errors on any
# cmake_minimum_required below 3.5 -- same fix as fast_livo2's Dockerfile,
# needed for the identical from-source Livox dependency below.
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

WORKDIR /workspace

# Livox-SDK2 (system-wide), then livox_ros_driver2 as a colcon package --
# from-source, same mechanism as fast_livo2's Dockerfile (see that file's
# header comment for why this wasn't swapped for a conda-forge/
# robostack-jazzy package: unconfirmed header-for-header compatibility,
# prefer the already-verified from-source approach).
RUN /bin/bash -c "source /pixi-project/activate.sh && \
    git clone --depth 1 https://github.com/Livox-SDK/Livox-SDK2.git && \
    mkdir Livox-SDK2/build && cd Livox-SDK2/build && cmake .. && make -j\"\$(nproc)\" && make install"

RUN mkdir -p /colcon_ws/src
RUN git clone --depth 1 https://github.com/Livox-SDK/livox_ros_driver2.git /colcon_ws/src/livox_ros_driver2 && \
    cd /colcon_ws/src/livox_ros_driver2 && \
    cp package_ROS2.xml package.xml && \
    cp -r launch_ROS2/ launch/

COPY third_party/slam/gs_livo /colcon_ws/src/gs_livo

# rpg_vikit (vendored by gs_livo's own src/ tree, sub-fetched from raw
# Robotic-Developer-Road/rpg_vikit by `scripts/fetch_slam_backends.py`) is
# raw, unpatched upstream at this point -- overlay the patched
# ament_cmake-native versions (vikit_common's upstream IF(USE_ROS) catkin/
# plain-CMake fork had no ament support at all; vikit_ros had spurious
# rosidl deps) onto it. These patches were originally authored in-session
# directly against a local checkout and never had a permanent home (unlike
# fast_livo2's rpg_vikit, which gets small inline sed patches instead --
# gs_livo's rpg_vikit needed substantially more rework: a new top-level
# CMakeLists.txt plus rewritten vikit_common/vikit_ros CMakeLists.txt and
# package.xml, too much for sed one-liners to stay legible) -- committing
# them here as real files is what makes them reproducible for anyone
# re-running this Dockerfile from a clean third_party/ vendor.
COPY docker/slam/gs_livo/patches/rpg_vikit/CMakeLists.txt \
    /colcon_ws/src/gs_livo/src/rpg_vikit/CMakeLists.txt
COPY docker/slam/gs_livo/patches/rpg_vikit/vikit_common/CMakeLists.txt \
    /colcon_ws/src/gs_livo/src/rpg_vikit/vikit_common/CMakeLists.txt
COPY docker/slam/gs_livo/patches/rpg_vikit/vikit_common/package.xml \
    /colcon_ws/src/gs_livo/src/rpg_vikit/vikit_common/package.xml
COPY docker/slam/gs_livo/patches/rpg_vikit/vikit_ros/CMakeLists.txt \
    /colcon_ws/src/gs_livo/src/rpg_vikit/vikit_ros/CMakeLists.txt
COPY docker/slam/gs_livo/patches/rpg_vikit/vikit_ros/package.xml \
    /colcon_ws/src/gs_livo/src/rpg_vikit/vikit_ros/package.xml

# gs_livo vendors several header-only third-party libraries directly under
# src/ (eigen -- a full copy of Eigen's own source tree, glm, json, args,
# tinyply, simple-knn) plus the CUDA Gaussian-rasterizer lib3dgs/, all
# consumed via add_subdirectory()/include_directories() from gs-livo's own
# CMakeLists.txt -- none of them are meant to be independent colcon
# packages. Since each ships its own top-level CMakeLists.txt (Eigen's for
# its own test suite, lib3dgs's for its CUDA kernels), colcon's package
# discovery walks into every one of them and tries to configure+build them
# standalone, which fails outright (confirmed: "Failed <<< Eigen3
# [CMake Error at scripts/CMakeLists.txt:2 (configure_file)]" plus three
# more aborted). Same COLCON_IGNORE marker trick already used at
# third_party/COLCON_IGNORE to keep colcon out of vendored SLAM sources
# entirely, applied per-subdirectory here since gs-livo's own build still
# needs colcon to walk INTO its own directory, just not into these siblings.
# rpg_vikit itself also needs the marker: colcon would otherwise discover
# its top-level wrapper CMakeLists.txt (project(rpg_vikit_wrapper)) as YET
# ANOTHER independent package and try to build vikit_common/vikit_ros
# standalone in a separate configure run, where vikit_ros's own
# find_package(vikit_common) fails outright since nothing is installed yet
# within that separate run -- confirmed by hitting exactly this. gs-livo's
# own add_subdirectory(rpg_vikit) already builds both in-tree, as the one
# and only build of them that actually needs to happen.
RUN for d in eigen glm json args tinyply simple-knn lib3dgs rpg_vikit; do \
        touch "/colcon_ws/src/gs_livo/src/$d/COLCON_IGNORE"; \
    done

# gs-livo/CMakeLists.txt's LibTorch discovery (Torch_DIR pointed at a
# manually-downloaded-and-extracted "./libtorch" zip layout, not how the
# pixi/conda-forge `libtorch` package installs) is fixed directly in the
# fork's committed source (commit b5ec0a3: Torch_DIR now resolves via
# $ENV{CONDA_PREFIX}/share/cmake/Torch) -- no Dockerfile-level patch
# needed here anymore.

# tinyply -- not on conda-forge, stays a from-source build. Small,
# self-contained, ships its own working CMakeLists.txt/Config.cmake.in --
# build and install it system-wide (like Livox-SDK2) so lib3dgs's
# find_package(tinyply REQUIRED) resolves normally.
RUN /bin/bash -c "source /pixi-project/activate.sh && \
    cd /colcon_ws/src/gs_livo/src/tinyply && \
    mkdir build && cd build && cmake .. && make -j\"\$(nproc)\" && make install"

# lib3dgs (the CUDA Gaussian-rasterizer library gs-livo links against) has
# its own separate CMakeLists.txt and was never actually built by anything
# -- gs-livo's own CMakeLists.txt only references the pre-built
# lib3dgs/build/lib3dgs_lib.so output by absolute expectation, it never
# add_subdirectory()s or otherwise invokes lib3dgs's build. Confirmed by
# hitting "No rule to make target '.../lib3dgs/build/lib3dgs_lib.so'" at
# the final fastlivo_mapping link step. Build it standalone first, same
# pattern as Livox-SDK2/tinyply above, producing exactly the artifact path
# gs-livo's own CMakeLists.txt (LIB3DGS_LIB_PATH) expects -- not installed
# system-wide, just built in place.
# -j4, not -j"$(nproc)": nvcc's per-translation-unit memory footprint for
# CUDA kernels this size is large enough that full-core parallelism here
# risks exhausting host RAM (observed: repeated unexplained build
# terminations mid-compile on a 16-core/15GB host, consistent with an OOM
# kill, not a real error -- every other build step in this Dockerfile
# builds fine at full parallelism, only this CUDA compilation needed
# capping).
RUN /bin/bash -c "source /pixi-project/activate.sh && \
    export CC=\$(which x86_64-conda-linux-gnu-gcc) CXX=\$(which x86_64-conda-linux-gnu-g++) && \
    cd /colcon_ws/src/gs_livo/src/lib3dgs && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_CUDA_HOST_COMPILER=\$CXX && \
    make -j4"

WORKDIR /colcon_ws
# CC/CXX pinned to the pixi env's GCC 12 toolchain (see header comment) --
# CUDA 12.x's nvcc rejects newer host compilers, and this must be set for
# BOTH the plain C++ packages and nvcc's own host-compiler selection.
# -DROS_EDITION/-DDISTRO_ROS are livox_ros_driver2's own CMakeLists.txt
# switches -- without DISTRO_ROS=jazzy it falls through to an older,
# broken get_target_property()-based lookup for its own generated message
# typesupport target instead of the correct modern
# rosidl_get_typesupport_target() path, failing with "LIVOX_INTERFACES_
# INCLUDE_DIRECTORIES ... set to NOTFOUND" -- confirmed by hitting exactly
# this (fast_livo2's Dockerfile already passes these same two args for the
# identical reason; this one was missed on the first pass here).
RUN /bin/bash -c "source /pixi-project/activate.sh && \
    export CC=\$(which x86_64-conda-linux-gnu-gcc) CXX=\$(which x86_64-conda-linux-gnu-g++) && \
    colcon build --symlink-install --cmake-args -DCMAKE_CUDA_HOST_COMPILER=\$CXX -DROS_EDITION=ROS2 -DDISTRO_ROS=jazzy"

COPY docker/slam/shared/start_slam_backend.sh /start_slam_backend.sh
RUN chmod +x /start_slam_backend.sh
ENTRYPOINT ["/start_slam_backend.sh"]
