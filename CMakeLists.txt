cmake_minimum_required(VERSION 3.21)
project(opencl_cook)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIR})
link_directories(${OpenCL_LIBRARY})
add_executable(array_add array_add/array_add.cpp)
target_link_libraries(array_add "-framework OpenCL")
