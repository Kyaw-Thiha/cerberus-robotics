cmake_minimum_required(VERSION 3.8)
project(cerberus_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TaggedMap.msg"
  "msg/LocoCommand.msg"
  "msg/ArmCommand.msg"
  "msg/BodyCommand.msg"
  "msg/HeightScan.msg"
  DEPENDENCIES sensor_msgs std_msgs geometry_msgs
)

ament_package()
