cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

# add_executable(each_cpp_forwarder each_cpp_forwarder.cpp)
add_library(each_cpp_forwarder SHARED each_cpp_forwarder.cpp)
target_link_libraries(each_cpp_forwarder "${TORCH_LIBRARIES}")
set_property(TARGET each_cpp_forwarder PROPERTY CXX_STANDARD 14)
