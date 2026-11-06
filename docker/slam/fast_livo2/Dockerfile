# Cerberus -- SLAM backend image: FAST-LIVO2 (LiDAR+camera baseline).
#
# ROS2 Jazzy -- CONFIRMED building clean end-to-end (2026-09-21, commit
# d5b7583 on the vendored fork Kyaw-Thiha/cerberus-FAST-LIVO2, a fork of
# Robotic-Developer-Road/FAST-LIVO2's Humble port). No ros1_bridge needed.
#
# This is the hardest of the 6 SLAM backends to convert to pixi because,
# beyond fast_livo itself, it built THREE extra things from source under
# the old apt-based image: Sophus 1.22.10 (patched), Livox-SDK2, and
# livox_ros_driver2 -- plus the Robotic-Developer-Road fork of rpg_vikit,
# which stays from-source regardless (see below). Each was re-investigated
# individually rather than assuming a blanket "convert everything":
#
# - Sophus: CONVERTED to pixi (conda-forge's `sophus` package, the
#   `slam-sophus` feature in root pixi.toml). `pixi search -c
#   robostack-jazzy -c conda-forge sophus` shows conda-forge does NOT carry
#   the 1.22.10 tag this fork's README pins -- only 22.04.1, 22.10 and
#   1!1.24.6 exist, so this is a minor-version bump (1.22.10 -> 1.24.6),
#   not an exact-version match. Verified safe by actually attempting the
#   build (not just assumed): fast_livo2's own CMakeLists.txt (lines
#   113-122) already handles modern Sophus's target-based CMake config via
#   a `TARGET Sophus::Sophus` fallback for the legacy Sophus_INCLUDE_DIRS/
#   Sophus_LIBRARIES variables, and the fork's actual Sophus usage
#   (`using namespace Sophus`, `SE3<double>` in common_lib.h/frame.h/
#   feature.h/vio.h/vio.cpp/LIVMapper.cpp) is limited to the stable
#   SE3/SO3 Lie-group API -- no exotic/removed symbols. This drops the
#   from-source Sophus clone + FindEigen3.cmake sed-patch step entirely.
#
# - Livox-SDK2 and livox_ros_driver2: KEPT from-source, unchanged
#   mechanism, just now running inside the pixi-provisioned runtime image
#   instead of the apt-based one. Investigated `ros-jazzy-livox-ros-driver2`
#   (robostack-jazzy, v1.2.4 -- a metapackage wrapping `ros2-livox-ros-
#   driver2` v1.2.4) and `livox-sdk2` (conda-forge, v1.3.1/1.2.5) as
#   candidates to replace these. The actual code only depends on the
#   generated ROS2 message type `livox_ros_driver2::msg::CustomMsg`
#   (LIVMapper.cpp/preprocess.cpp) -- fast_livo2 also vendors an unused
#   dead ROS1 `livox_ros_driver/CustomMsg.h` under include/ that is never
#   included by any .cpp/.h in the tree (confirmed by grep), so it doesn't
#   affect this decision either way. The Livox CustomMsg/CustomPoint wire
#   format has been stable for years, so the conda packages are PLAUSIBLE
#   drop-ins, but: (a) the pinned Livox-SDK2 version the from-source
#   livox_ros_driver2 actually links against was never itself pinned
#   (plain `git clone` of the default branch, both before and after this
#   conversion) and (b) whether robostack-jazzy's `ros2-livox-ros-driver2`
#   build matches this exact fork's expectations header-for-header was not
#   independently confirmed by actually building against it. Per this
#   task's own guidance -- correctness matters more than eliminating every
#   possible source-build step, prefer the existing from-source approach
#   when in doubt -- these stay exactly as they were, just relocated to run
#   after the pixi ROS2 layer is activated instead of after apt's.
#
# - rpg_vikit (Robotic-Developer-Road fork, NOT the ROS1-only uzh-rpg
#   original): stays from-source -- confirmed NOT on conda-forge, a small
#   non-mainstream fork with no reason to expect it ever will be.
#
# ROS2 + cv_bridge/OpenCV + PCL + Boost + fmt/jsoncpp + Sophus are
# provisioned via pixi (the `slam-fast-livo2` environment in the repo's
# root pixi.toml, robostack-jazzy + conda-forge channels) instead of
# `apt-get install ros-jazzy-*` -- single reproducible dependency source of
# truth shared with local dev. First stage resolves+installs that pixi
# environment from the locked root pixi.toml/pixi.lock; only the resulting
# .pixi/envs/slam-fast-livo2 env and a shell-hook activation script are
# copied into the slim runtime stage below, not the pixi tool itself or the
# full repo. The shared entrypoint (start_slam_backend.sh) sources
# /pixi-project/activate.sh automatically if present. The runtime stage
# still needs its own C/C++ toolchain + git for the from-source steps below
# (Livox-SDK2, livox_ros_driver2, rpg_vikit) -- the pixi env's
# cxx-compiler/c-compiler/make/binutils/cmake cover that; only `git` comes
# from apt since it's not part of the pixi env.
#
# Source for FAST-LIVO2 itself is vendored locally -- run
# `python scripts/fetch_slam_backends.py fast_livo2` before building.

FROM ghcr.io/prefix-dev/pixi:noble AS pixi-build

WORKDIR /pixi-project
COPY pixi.toml pixi.lock ./
RUN pixi install -e slam-fast-livo2 --locked

# `pixi shell-hook` emits the activation script the runtime stage sources
# instead of baking `pixi run` into every command below.
RUN pixi shell-hook -e slam-fast-livo2 --shell bash > /pixi-project/activate.sh

FROM ubuntu:noble AS runtime

ENV DEBIAN_FRONTEND=noninteractive
# ca-certificates is required alongside git: ubuntu:noble ships no CA store
# by default, so `git clone https://...` below fails with "server
# certificate verification failed. CAfile: none" without it (confirmed by
# hitting that exact failure, not assumed).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=pixi-build /pixi-project/.pixi/envs/slam-fast-livo2 /pixi-project/.pixi/envs/slam-fast-livo2
COPY --from=pixi-build /pixi-project/activate.sh /pixi-project/activate.sh

# The pixi env's cmake (4.4.3) is much newer than apt's old ROS-base image
# ever carried, and CMake >=4.0 hard-errors on any `cmake_minimum_required`
# below 3.5 ("Compatibility with CMake < 3.5 has been removed"). Confirmed
# by hitting this exactly: Livox-SDK2's own CMakeLists.txt pins
# `VERSION 3.0`. CMAKE_POLICY_VERSION_MINIMUM (CMake >=3.31) is the
# documented escape hatch -- treat anything below 3.5 as if it asked for
# 3.5 instead of hard-failing. Needed for the from-source builds below
# (Livox-SDK2 directly; potentially livox_ros_driver2/rpg_vikit via
# colcon's own cmake invocation too), not for anything pixi/conda-managed.
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

WORKDIR /workspace

# Livox-SDK2 (system-wide), then livox_ros_driver2 as a colcon package --
# from-source, unchanged from the apt-based image (see header comment for
# why this wasn't swapped for conda-forge's livox-sdk2 / robostack-jazzy's
# ros-jazzy-livox-ros-driver2). The pixi env supplies the CMake/compiler
# toolchain this plain (non-ROS) library's `cmake .. && make` needs.
RUN /bin/bash -c "source /pixi-project/activate.sh && \
    git clone --depth 1 https://github.com/Livox-SDK/Livox-SDK2.git && \
    mkdir Livox-SDK2/build && cd Livox-SDK2/build && cmake .. && make -j\"\$(nproc)\" && make install"

RUN mkdir -p /colcon_ws/src
# livox_ros_driver2's upstream repo ships both ROS1 and ROS2 support from
# one tree, selected via its own build.sh: package_ROS1.xml/package_ROS2.xml
# instead of a plain package.xml, and launch_ROS2/ instead of launch/. A
# bare clone (as this step used to be) has no package.xml at all and fails
# colcon's package discovery -- confirmed by hitting exactly that. Do what
# build.sh does for the ROS2/jazzy path (its `colcon build` call, not
# `catkin_make`) by hand instead of shelling out to it, since it also wants
# to `rm -rf` the whole workspace's build/install dirs, which we don't want
# here.
RUN git clone --depth 1 https://github.com/Livox-SDK/livox_ros_driver2.git /colcon_ws/src/livox_ros_driver2 && \
    cd /colcon_ws/src/livox_ros_driver2 && \
    cp package_ROS2.xml package.xml && \
    cp -r launch_ROS2/ launch/

# rpg_vikit -- the Robotic-Developer-Road ROS2/Jazzy fork, NOT the uzh-rpg
# original (which is ROS1-only and does not build against Jazzy).
RUN git clone --depth 1 https://github.com/Robotic-Developer-Road/rpg_vikit.git /colcon_ws/src/rpg_vikit

# vikit_common/CMakeLists.txt unconditionally appends -std=c++0x (GCC) /
# -std=c++11, which as the LAST -std= flag on the compile command wins over
# anything else. conda-forge's Sophus 1.24.6 headers need C++17
# (std::optional, used directly in so3.hpp) -- confirmed by hitting this
# incrementally on a real build: c++11/c++0x failed on std::enable_if_t
# (C++14), and c++14 (tried first, insufficient) still failed on
# std::optional specifically (C++17). Patched here (not upstreamed -- this
# is Robotic-Developer-Road's fork, not one of ours) rather than swapping
# in the newer conda-forge Sophus's own headers, since this is the more
# targeted fix and doesn't touch the already-verified Sophus version
# decision documented above.
RUN sed -i 's/-std=c++0x/-std=c++17/; s/-std=c++11/-std=c++17/' \
    /colcon_ws/src/rpg_vikit/vikit_common/CMakeLists.txt

# vikit_common's own CMakeLists.txt has two compounding real bugs in the
# block that generates vikit_commonConfig.cmake for downstream
# find_package() consumers:
# 1. `SET(vikit_common_LIBRARIES ${FULL_LIBRARY_NAME})` -- FULL_LIBRARY_NAME
#    is never assigned anywhere (the GET_TARGET_PROPERTY() call that used
#    to set it is commented out immediately above), so
#    vikit_common_LIBRARIES silently ends up empty.
# 2. The line meant to replace it, `SET(VIKIT_COMMON_LOCATION
#    $<TARGET_FILE:vikit_common>)`, is a generator expression -- those only
#    resolve inside the SAME CMake configure context where the
#    `vikit_common` target actually exists. Baked into a Config.cmake
#    string for an entirely separate downstream project's `find_package()`
#    (vikit_ros, or fast_livo2 itself), there is no such target and CMake
#    errors outright: "Error evaluating generator expression:
#    $<TARGET_FILE:vikit_common> -- No target 'vikit_common'".
# Confirmed both by hitting them in sequence on real builds: first
# find_package(vikit_common) silently succeeding with an empty
# _LIBRARIES (linker: "undefined reference to vk::camera_loader::
# loadFromRosNs / vk::PinholeCamera::... / vk::shiTomasiScore /
# vk::halfSample"), then after fixing that, vikit_ros's own configure step
# failing outright on the generator-expression error above. Fix: compute a
# real, portable, configure-time (not generator-expression) path instead --
# the library installs to ${CMAKE_INSTALL_PREFIX}/lib per this same file's
# own INSTALL(TARGETS ...) call a few lines below.
RUN sed -i 's|set(VIKIT_COMMON_LOCATION \$<TARGET_FILE:vikit_common>)|set(VIKIT_COMMON_LOCATION "${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}vikit_common${CMAKE_SHARED_LIBRARY_SUFFIX}")|; s/SET(vikit_common_LIBRARIES \${FULL_LIBRARY_NAME} )/SET(vikit_common_LIBRARIES \${VIKIT_COMMON_LOCATION} )/' \
    /colcon_ws/src/rpg_vikit/vikit_common/CMakeLists.txt

# vikit_ros has the analogous bug at the ament_cmake layer: it does
# `install(TARGETS vikit_ros EXPORT vikit_ros ...)`, creating an export set
# named "vikit_ros", but never calls `ament_export_targets(vikit_ros)` to
# actually register that export set with ament_package()'s generated
# config -- so the export set is orphaned, and the compiled libvikit_ros.so
# (which is what vk::camera_loader::loadFromRosNs, used directly in
# LIVMapper.cpp, lives in) never reaches downstream find_package(vikit_ros)
# consumers. Confirmed by hitting exactly this as the last remaining
# undefined reference once vikit_common's linking was fixed above. Fix:
# add the missing ament_export_targets() call, the standard ament_cmake
# idiom for exporting a compiled library alongside an EXPORT set.
RUN sed -i 's/ament_export_include_directories(include)/ament_export_include_directories(include)\nament_export_targets(${PROJECT_NAME} HAS_LIBRARY_TARGET)/' \
    /colcon_ws/src/rpg_vikit/vikit_ros/CMakeLists.txt

# Adding the export set above makes CMake strictly validate every exported
# target's INTERFACE_INCLUDE_DIRECTORIES -- and vikit_ros's own
# target_include_directories() call adds a bare (non-generator-expression)
# ${CMAKE_CURRENT_SOURCE_DIR} alongside the already-correctly-wrapped
# $<BUILD_INTERFACE:...>/$<INSTALL_INTERFACE:...> pair right above it,
# which trips that validation outright: "Target vikit_ros
# INTERFACE_INCLUDE_DIRECTORIES property contains path ... which is
# prefixed in the source directory." Confirmed by hitting this immediately
# after the ament_export_targets fix above. Wrap it in the same
# $<BUILD_INTERFACE:...> guard as its neighboring line -- it's a source-tree
# path only meaningful while building vikit_ros itself, not something a
# downstream installed consumer needs.
RUN sed -i 's|^  \${CMAKE_CURRENT_SOURCE_DIR}  # Add this line to include the project root directory|  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>  # Add this line to include the project root directory|' \
    /colcon_ws/src/rpg_vikit/vikit_ros/CMakeLists.txt

COPY third_party/slam/fast_livo2 /colcon_ws/src/fast_livo2

WORKDIR /colcon_ws
# -DROS_EDITION/-DDISTRO_ROS are livox_ros_driver2's own CMakeLists.txt
# switches (see its build.sh) for selecting the ROS2 code path; harmless
# unused cache vars for the other packages in this workspace.
RUN /bin/bash -c "source /pixi-project/activate.sh && colcon build --symlink-install --packages-up-to fast_livo --cmake-args -DROS_EDITION=ROS2 -DDISTRO_ROS=jazzy"

COPY docker/slam/shared/start_slam_backend.sh /start_slam_backend.sh
RUN chmod +x /start_slam_backend.sh
ENTRYPOINT ["/start_slam_backend.sh"]
