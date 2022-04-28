#
#   Copyright (c) 2022, The 510 Research Group, College of Engineering, Peking University
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#	   http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#   Author: Denghui Lu <denghuilu@pku.edu.cn>, 
#       The 510 Research Group,
#		   College of Engineering, Peking University
#

cmake_minimum_required(VERSION 3.5)
project(gpu_programming_samples_test)

enable_testing()

set(CMAKE_LINK_WHAT_YOU_USE TRUE)

set(libname "common")
set(LIB_BASE_DIR ${CMAKE_SOURCE_DIR}/../)

include_directories(${LIB_BASE_DIR}/include)
file(GLOB LIB_SRC ${LIB_BASE_DIR}/src/*.cc ${LIB_BASE_DIR}/src/*.cpp)
message(status ${LIB_SRC})
# add_library(${libname} SHARED ${LIB_SRC})
add_library(${libname} ${LIB_SRC})

message(status "${CMAKE_SOURCE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Devices that have both ROCM and CUDA are not currently supported
if (USE_ROCM_TOOLKIT AND USE_CUDA_TOOLKIT)
  message (FATAL_ERROR "Devices that have both ROCM and CUDA are not currently supported")
endif()

# define USE_CUDA_TOOLKIT
if (USE_CUDA_TOOLKIT)
  find_package(CUDA REQUIRED)
  add_definitions("-DUSE_CUDA_TOOLKIT")
  message(STATUS "Found CUDA in ${CUDA_TOOLKIT_ROOT_DIR}, build nv GPU support")
  include_directories(${CUDA_INCLUDE_DIRS})
  add_subdirectory(${LIB_BASE_DIR}/src/cuda cuda_binary_dir)
else()
  message(STATUS "Will not build nv GPU support")
endif(USE_CUDA_TOOLKIT)

#define USE_ROCM_TOOLKIT
if (USE_ROCM_TOOLKIT)
  list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake/)
  find_package(ROCM REQUIRED)
  add_definitions("-DUSE_ROCM_TOOLKIT")
  add_compile_definitions(__HIP_PLATFORM_HCC__)
  include_directories(${ROCM_INCLUDE_DIRS})
  add_subdirectory(${LIB_BASE_DIR}/src/rocm rocm_binary_dir)
else()
  message(STATUS "Will not build AMD GPU support")
endif (USE_ROCM_TOOLKIT)

file(GLOB TEST_SRC test_*.cc)
add_executable( runUnitTests ${TEST_SRC} )

if (USE_CUDA_TOOLKIT)
  target_link_libraries(runUnitTests gtest gtest_main ${libname} pthread lib_cuda)
elseif (USE_ROCM_TOOLKIT)
  target_link_libraries(runUnitTests gtest gtest_main ${libname} pthread lib_rocm ${ROCM_LIBRARIES})
else()
  target_link_libraries(runUnitTests gtest gtest_main ${libname} pthread)
endif()
add_test(runUnitTests runUnitTests )

find_package(GTest)
if(NOT GTEST_LIBRARIES)
  configure_file(../../cmake/google_test.cmake.in googletest-download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
  if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
  if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
  endif()
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src ${CMAKE_CURRENT_BINARY_DIR}/googletest-build EXCLUDE_FROM_ALL)
else ()
  include_directories(${GTEST_INCLUDE_DIRS})
endif ()

install(TARGETS runUnitTests DESTINATION bin/)