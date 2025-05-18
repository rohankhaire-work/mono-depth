cmake_minimum_required(VERSION 3.15)
project(mono_depth)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=20012")
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(TensorRT REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(spdlog REQUIRED)
find_package(pcl_ros REQUIRED)


add_executable(mono_depth src/depth_estimation_node.cpp                           src/depth_estimation.cpp)
target_include_directories(mono_depth PRIVATE include
                                            ${OpenCV_INCLUDE_DIRS}
                                            ${CUDAToolkit_INCLUDE_DIR} 
                                            ${TensorRT_INCLUDE_DIRS})
ament_target_dependencies(mono_depth rclcpp cv_bridge image_transport pcl_ros tf2_ros)

target_link_libraries(mono_depth
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    Eigen3::Eigen
    ${CUDAToolkit_LIBRARY_DIR}/libcudart.so
    ${TensorRT_LIBRARY_DIRS}/libnvinfer.so)

install(TARGETS mono_depth
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY config launch weights
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
