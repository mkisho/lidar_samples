cmake_minimum_required(VERSION 3.0.2)
project(lidar_samples)

find_package(catkin REQUIRED COMPONENTS
  gazebo_msgs
  geometry_msgs
  roscpp
  sensor_msgs
  std_msgs
  roslib
)

catkin_package(
#     INCLUDE_DIRS include CATKIN_DEPENDS roscpp std_msgs roslib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

 add_executable(${PROJECT_NAME}_node src/lidar_samples.cpp)
 target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
 )

catkin_install_python(PROGRAMS src/detector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
