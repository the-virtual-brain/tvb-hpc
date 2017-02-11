cmake_minimum_required(VERSION 3.1.0)
project(tvb-hpc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -mavx2 -O3 -fopenmp -g")
include_directories("${CMAKE_CURRENT_LIST_DIR}/include")

add_subdirectory(extern/gbench)
#add_subdirectory(extern/gtest/googletest)

include_directories("${CMAKE_CURRENT_LIST_DIR}/extern/gbench/include")
#include_directories("${CMAKE_CURRENT_LIST_DIR}/extern/gtest/googletest/include")

find_package(Boost REQUIRED system)
include_directories(${Boost_INCLUDE_DIRS})

#find_package(Hdf5 REQUIRED)
#include_directories(${Hdf5_INCLUDE_DIRS})

add_executable(tests 
    src/test_euler.cc
    src/test_hmje.cc
    src/test_main.cc
    src/test_rww.cc
    src/test_util.cc
    )

add_executable(benchmarks src/bm_model_step.cc)
target_link_libraries(benchmarks benchmark ${Boost_LIBRARIES})
