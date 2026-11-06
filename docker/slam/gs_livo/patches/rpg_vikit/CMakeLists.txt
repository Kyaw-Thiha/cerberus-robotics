# Wrapper for the vendored rpg_vikit tree (Robotic-Developer-Road/rpg_vikit,
# ROS2 Jazzy port). The upstream repo has NO top-level CMakeLists.txt --
# vikit_common and vikit_ros are intended as sibling ament packages. In the
# gs-livo vendoring layout they live under add_subdirectory(rpg_vikit), so
# this wrapper exists only to descend into them. Both subtrees are patched
# to be pure ament_cmake packages (vikit_common needed a non-trivial patch
# because upstream's CMakeLists.txt is structurally catkin-or-plain-cmake,
# not ament_cmake; vikit_ros was already correctly ament_cmake upstream).
cmake_minimum_required(VERSION 3.16)
project(rpg_vikit_wrapper NONE)

add_subdirectory(vikit_common)
add_subdirectory(vikit_ros)
