cmake_minimum_required(VERSION 3.16)
project(vikit_ros VERSION 0.0.0 LANGUAGES CXX)

# Patched from upstream: dropped the spurious ${cpp_typesupport_target}
# reference (variable was never defined upstream either -- the original
# targets rclcpp::typesupport_cpp via ament_target_dependencies, which is
# the only correct ament_cmake way to expose that). Also dropped the
# ament_export_dependencies(rosidl_default_runtime) call -- this package
# does not generate any .msg/.srv/.idl files (verified: src/ contains
# only output_helper.cpp + camera_loader.cpp, no .msg/.srv), so it must
# not pull in rosidl_default_generators / rosidl_default_runtime. The
# matching package.xml edits drop the corresponding <buildtool_depend>,
# <exec_depend>, and <member_of_group> entries.

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -fexceptions")

message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  include(ProcessorCount)
  ProcessorCount(N)
  message(STATUS "CPU cores: ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN -DMP_PROC_NUM=3)
  elseif(N GREATER 3)
    add_definitions(-DMP_EN -DMP_PROC_NUM=2)
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Dependencies (note: vikit_common must already be built -- either as an
# already-installed, separate ament package find_package() can locate
# normally, or as an in-tree CMake target already defined by a sibling
# add_subdirectory(vikit_common) call in the same configure run, as gs-livo
# does via add_subdirectory(rpg_vikit). Only call find_package() in the
# former case -- calling it when the target already exists in-tree fails,
# since nothing has been installed yet within that single configure run.)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
if(NOT TARGET vikit_common)
  find_package(vikit_common REQUIRED)
endif()
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# vikit_common deliberately excluded from this list -- ament_target_dependencies
# resolves entries via find_package()-populated variables/imported targets,
# which the in-tree add_subdirectory(vikit_common) case never populates.
# vikit_common's own CMakeLists.txt exports its target unnamespaced
# (install(TARGETS vikit_common EXPORT export_vikit_common ...), no
# NAMESPACE), so the same plain `vikit_common` target name resolves
# correctly whether it came from find_package() or an in-tree sibling
# add_subdirectory() -- link it directly below instead.
set(dependencies
  rclcpp
  visualization_msgs
  tf2_ros
  tf2
  Eigen3
  tf2_geometry_msgs
)

# Source list.
list(APPEND SOURCEFILES
  src/output_helper.cpp
  src/camera_loader.cpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCEFILES})
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  ${Sophus_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} vikit_common)

ament_target_dependencies(${PROJECT_NAME} ${dependencies})

install(DIRECTORY include/
  DESTINATION include
  COMPONENT ${PROJECT_NAME}
)

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

ament_export_include_directories(include)
ament_export_targets(${PROJECT_NAME})
ament_export_dependencies(${dependencies})

ament_package()
