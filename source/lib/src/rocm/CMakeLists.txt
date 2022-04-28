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

# required cmake version
cmake_minimum_required(VERSION 3.5)
# project name
project(lib_rocm)
set(CMAKE_LINK_WHAT_YOU_USE TRUE)

# set c++ version c++11
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_HIP_STANDARD 11)

message(STATUS "HIP major version is " ${HIP_VERSION_MAJOR})

set (HIP_HIPCC_FLAGS -hc; -fno-gpu-rdc; --amdgpu-target=gfx906; -fPIC; -O3; --std=c++11)

file (GLOB SOURCE_FILES "*.hip.cu" )

hip_add_library(lib_rocm SHARED ${SOURCE_FILES})
target_include_directories(lib_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include/)

install(TARGETS lib_rocm DESTINATION lib/)