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

# library name
set (libname ${LIB_NAME})

file(GLOB LIB_SRC src/*.cpp)
file(GLOB INC_SRC include/*.h)

add_library(${libname} SHARED ${LIB_SRC})
target_include_directories(${libname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Devices that have both ROCM and CUDA are not currently supported
if (USE_ROCM_TOOLKIT AND USE_CUDA_TOOLKIT)
  message (FATAL_ERROR "Devices that have both ROCM and CUDA are not currently supported")
endif()

if (USE_CUDA_TOOLKIT)
  add_subdirectory(src/cuda)
  add_definitions("-DUSE_CUDA_TOOLKIT")
  set (EXTRA_LIBS ${EXTRA_LIBS} lib_cuda)
  target_link_libraries (${libname} PUBLIC ${CUDA_LIBRARIES} ${EXTRA_LIBS})
  # gpu_cuda.h
  target_include_directories(${libname} PUBLIC ${CUDA_INCLUDE_DIRS})
endif()

if (USE_ROCM_TOOLKIT)
  add_subdirectory(src/rocm)
  add_definitions("-DUSE_ROCM_TOOLKIT")
  set (EXTRA_LIBS ${EXTRA_LIBS} lib_rocm)
  target_link_libraries (${libname} PUBLIC ${ROCM_LIBRARIES} ${EXTRA_LIBS})
  # gpu_rocm.h
  target_include_directories(${libname} PUBLIC ${ROCM_INCLUDE_DIRS})
endif()

set_target_properties(${libname} PROPERTIES INSTALL_RPATH $ORIGIN)