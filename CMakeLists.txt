cmake_minimum_required(VERSION 3.1.0)
project(tvb-hpc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories("${CMAKE_CURRENT_LIST_DIR}/include")

add_executable(tests 
    src/test_euler.cc
    src/test_hmje.cc
    src/test_main.cc
    src/test_rww.cc
    src/test_util.cc
    )
