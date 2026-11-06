# Patched from upstream Robotic-Developer-Road/rpg_vikit for ament_cmake.
# Upstream was structured as IF(USE_ROS) catkin ELSE() plain CMake with no
# ament_cmake support -- this rewrite drops the catkin branch entirely
# (gs-livo is ROS2-only) and uses ament idioms (target_link_libraries on
# modern imported targets, ament_export_targets, ament_export_dependencies,
# ament_package). C++ source list, headers, and tests preserved verbatim.

cmake_minimum_required(VERSION 3.16)
project(vikit_common VERSION 0.0.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Platform-specific flags (preserved from upstream).
if(DEFINED ENV{ARM_ARCHITECTURE})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -march=armv8-a")
endif()
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
endif()
if(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

# Plain CMake (system) packages.
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)

# ament_cmake glue.
find_package(ament_cmake REQUIRED)

# Source list (preserved from upstream).
list(APPEND SOURCEFILES
  src/atan_camera.cpp
  src/omni_camera.cpp
  src/math_utils.cpp
  src/vision.cpp
  src/performance_monitor.cpp
  src/robust_cost.cpp
  src/user_input_thread.cpp
  src/pinhole_camera.cpp
  src/equidistant_camera.cpp
  src/polynomial_camera.cpp
  src/homography.cpp
  src/img_align.cpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCEFILES})
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${OpenCV_LIBRARIES}
    Sophus::Sophus
    fmt::fmt
)

# Tests (preserved from upstream). ament_register_gtest wraps the legacy
# ADD_EXECUTABLE form so they build under colcon, with --build-test on.
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_vk_common_camera test/test_camera.cpp)
  target_link_libraries(test_vk_common_camera ${PROJECT_NAME} ${OpenCV_LIBRARIES})

  ament_add_gtest(test_vk_common_triangulation test/test_triangulation.cpp)
  target_link_libraries(test_vk_common_triangulation ${PROJECT_NAME} ${OpenCV_LIBRARIES})

  ament_add_gtest(test_vk_common_patch_score test/test_patch_score.cpp)
  target_link_libraries(test_vk_common_patch_score ${PROJECT_NAME} ${OpenCV_LIBRARIES})
endif()

# Install headers + library.
install(DIRECTORY include/vikit DESTINATION include)

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(Eigen3 Sophus fmt OpenCV)

ament_package()
