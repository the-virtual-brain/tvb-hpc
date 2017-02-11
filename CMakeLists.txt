cmake_minimum_required(VERSION 3.1.0)
project(tvb-hpc)

if("${CMAKE_CXX_COMPILER}" STREQUAL "pgc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --c++11")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")

#find_package(Hdf5 REQUIRED)
#include_directories(${Hdf5_INCLUDE_DIRS})

add_library(sim_simple SHARED src/sim_simple.cc)
