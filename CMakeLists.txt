cmake_minimum_required(VERSION 3.1.0)
project(tvb-hpc)

add_subdirectory(extern/pybind11)
include_directories("${CMAKE_CURRENT_LIST_DIR}/include")

add_subdirectory("src")
