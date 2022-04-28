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

#include "lib.h"
#include "device.h"

template<typename FPTYPE> 
void init_data(
    const int flag,
    const int dim,
    const std::string device,
    FPTYPE * &A,
    FPTYPE * &B,
    FPTYPE * &C
)
{
  int size = 0;

  switch (flag) {
    case 1:
      size = dim;
      break;
    case 2:
      size = dim * dim;
      break;
    default:
      break;
  }

  #if USE_CUDA_TOOLKIT || USE_ROCM_TOOLKIT
  // allocate CPU memory.
  FPTYPE * h_A = NULL, * h_B = NULL, * h_C = NULL;
  h_A = (FPTYPE *)malloc(sizeof(FPTYPE) * size);
  h_B = (FPTYPE *)malloc(sizeof(FPTYPE) * size);
  h_C = (FPTYPE *)malloc(sizeof(FPTYPE) * size);
  memset(h_A, 0.0, sizeof(FPTYPE) * size);
  memset(h_B, 0.0, sizeof(FPTYPE) * size);
  memset(h_C, 0.0, sizeof(FPTYPE) * size);
  for (int ii = 0; ii < size; ii++) {
    h_A[ii] = 1.0;
    h_B[ii] = 2.0;
  }
  // allocate Device memory.
  malloc_device_memory(A, size);
  malloc_device_memory(B, size);
  malloc_device_memory(C, size);
  memcpy_host_to_device(A, h_A, size);
  memcpy_host_to_device(B, h_B, size);
  memcpy_host_to_device(C, h_C, size);
  free(h_A);
  free(h_B);
  free(h_C);
  #else
  A = (FPTYPE *)malloc(sizeof(FPTYPE) * size);
  B = (FPTYPE *)malloc(sizeof(FPTYPE) * size);
  C = (FPTYPE *)malloc(sizeof(FPTYPE) * size);
  memset(A, 0.0, sizeof(FPTYPE) * size);
  memset(B, 0.0, sizeof(FPTYPE) * size);
  memset(C, 0.0, sizeof(FPTYPE) * size);
  for (int ii = 0; ii < size; ii++) {
    A[ii] = 1.0;
    B[ii] = 2.0;
  }
  #endif
}

template<typename FPTYPE> 
void delete_data(
    const int flag,
    const int dim,
    const std::string device,
    FPTYPE * &A,
    FPTYPE * &B,
    FPTYPE * &C
)
{
  #if USE_CUDA_TOOLKIT || USE_ROCM_TOOLKIT
  // allocate CPU memory.
  delete_device_memory(A);
  delete_device_memory(B);
  delete_device_memory(C);
  #else
  free(A);
  free(B);
  free(C);
  #endif
}

template<typename FPTYPE> 
void vector_add_cpu(
    const int size,
    const FPTYPE * A,
    const FPTYPE * B,
    FPTYPE * C)
{
  for (int ii = 0; ii < size; ii++) {
    C[ii] = A[ii] + B[ii];
  }
}

template<typename FPTYPE> 
void matrix_add_cpu(
    const int row,
    const int col,
    const FPTYPE * A,
    const FPTYPE * B,
    FPTYPE * C)
{
  for (int ii = 0; ii < row; ii++) {
    for (int jj = 0; jj < col; jj++) {
      C[ii * col + jj] = A[ii * col + jj] + B[ii * col + jj];
    }
  }
}

template void init_data<int>(const int flag, const int dim, std::string device, int * &A, int * &B, int * &C);
template void init_data<float>(const int flag, const int dim, std::string device, float * &A, float * &B, float * &C);
template void init_data<double>(const int flag, const int dim, std::string device, double * &A, double * &B, double * &C);

template void delete_data<int>(const int flag, const int dim, std::string device, int * &A, int * &B, int * &C);
template void delete_data<float>(const int flag, const int dim, std::string device, float * &A, float * &B, float * &C);
template void delete_data<double>(const int flag, const int dim, std::string device, double * &A, double * &B, double * &C);


template void vector_add_cpu<int>(const int size, const int * A, const int * B, int * C);
template void vector_add_cpu<float>(const int size, const float * A, const float * B, float * C);
template void vector_add_cpu<double>(const int size, const double * A, const double * B, double * C);

template void matrix_add_cpu<int>(const int row, const int col, const int * A, const int * B, int * C);
template void matrix_add_cpu<float>(const int row, const int col, const float * A, const float * B, float * C);
template void matrix_add_cpu<double>(const int row, const int col, const double * A, const double * B, double * C);