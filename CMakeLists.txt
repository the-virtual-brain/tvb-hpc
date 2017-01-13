cmake_minimum_required(VERSION 2.8.2)
project(tvb-hpc)

enable_testing()

# Python
if (NOT DEFINED "python_config")
    message(STATUS "python_config not provided, assuming 'python-config'")
    set(python_config "python-config")
endif()

execute_process(COMMAND "${python_config}" "--prefix"
    OUTPUT_VARIABLE python_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Python prefix is '${python_prefix}'")

