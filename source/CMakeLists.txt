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

# Define names of libs
# The gps indicates the gpu programming samples
cmake_minimum_required(VERSION 3.5)

project(gpu_programming_samples)

# Compile the executable program
file(GLOB MAIN_SRC main/*.cpp)
add_executable(gpu_programming_samples ${MAIN_SRC})

# define USE_CUDA_TOOLKIT
if (USE_CUDA_TOOLKIT)
  find_package(CUDA REQUIRED)
  add_definitions("-DUSE_CUDA_TOOLKIT")
  message(STATUS "Found CUDA in ${CUDA_TOOLKIT_ROOT_DIR}, build nv GPU support")
else()
  message(STATUS "Will not build nv GPU support")
endif(USE_CUDA_TOOLKIT)

#define USE_ROCM_TOOLKIT
if (USE_ROCM_TOOLKIT)
  find_package(ROCM REQUIRED)
  add_definitions("-DUSE_ROCM_TOOLKIT")
  add_compile_definitions(__HIP_PLATFORM_HCC__)
  message(STATUS "Found ROCM in ${ROCM_ROOT}, build AMD GPU support")
else()
  message(STATUS "Will not build AMD GPU support")  
endif (USE_ROCM_TOOLKIT)

set (LIB_NAME	"common")
add_subdirectory(lib)

# Link the libraries according to the device conditions
if (USE_CUDA_TOOLKIT)
  target_link_libraries(gpu_programming_samples ${LIB_NAME} lib_cuda)
elseif (USE_ROCM_TOOLKIT)
  target_link_libraries(gpu_programming_samples ${LIB_NAME} lib_rocm ${ROCM_LIBRARIES})
else()
  target_link_libraries(gpu_programming_samples ${LIB_NAME})
endif()
