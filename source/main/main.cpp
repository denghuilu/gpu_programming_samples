/*
   Copyright (c) 2022, The 510 Research Group, College of Engineering, Peking University
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
	   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   Author: Denghui Lu <denghuilu@pku.edu.cn>, 
       The 510 Research Group,
		   College of Engineering, Peking University
*/

#include <vector>
#include <iostream> 
#include "lib.h"
#include "device.h"

#define DIM 10240

int get_computing_flag() {
  int config = 1;
  std::cout << "welcome to use the gpu_programming_samples system, please enter your choice:" << std::endl
            << "1: vectorAdd" << std::endl
            << "2: matrixAdd" << std::endl;
  std::cin >> config;
  return config;
}

// Return "GPU" device flag only if within the CUDA or ROCm environment 
// and the GPU device actually exists.
std::string get_device() {
  std::string device = "CPU";
  #if USE_CUDA_TOOLKIT || USE_ROCM_TOOLKIT
    if (get_device_count() >= 1) {
      device = "GPU";
    }
  #endif
  return device;
}

int main () {
  // Get the execution instruction
  const int flag = get_computing_flag();
  std::string device = get_device();
  float * A = NULL, * B = NULL, * C = NULL;
  init_data(
    flag, DIM, device,
    A, B, C);

  switch (flag) {
    case 1:
      if (device == "CPU") {
        vector_add_cpu(DIM, A, B, C);
      }
      else if (device == "GPU") {
        #if USE_CUDA_TOOLKIT
        vector_add_gpu_cuda(DIM, A, B, C);
        #elif USE_ROCM_TOOLKIT
        vector_add_gpu_rocm(DIM, A, B, C);
        #endif
      }
      break;
    case 2:
      if (device == "CPU") {
        matrix_add_cpu(DIM, DIM, A, B, C);
      }
      else if (device == "GPU") {
        #if USE_CUDA_TOOLKIT
        matrix_add_gpu_cuda(DIM, DIM, A, B, C);
        #elif USE_ROCM_TOOLKIT
        matrix_add_gpu_rocm(DIM, DIM, A, B, C);
        #endif
      }
      break;
    default:
      std::cout << "Error! input flag does not exist!" << std::endl;
      break;
  }

  // print_result(
  //   flag, DIM, device,
  //   A, B, C);

  delete_data(
    flag, DIM, device,
    A, B, C);
}