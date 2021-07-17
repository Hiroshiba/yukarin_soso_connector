cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add_executable(each_cpp_forwarder each_cpp_forwarder.cpp)
add_library(each_cpp_forwarder SHARED each_cpp_forwarder.cpp)
target_link_libraries(each_cpp_forwarder "${TORCH_LIBRARIES}")
set_property(TARGET each_cpp_forwarder PROPERTY CXX_STANDARD 14)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(
        TARGET each_cpp_forwarder
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:each_cpp_forwarder>
    )
endif (MSVC)
