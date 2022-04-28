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

include "lib.h"

template<
    typename FPTYPE,
    int      THREADS_PER_BLOCK> 
__global__ void vector_add_gpu_rocm_kernel(
    const int size,
    const FPTYPE * A,
    const FPTYPE * B,
    FPTYPE * C)
{
  const int thread_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (thread_idx > size) {
    return;
  }

  C[thread_idx] = A[thread_idx] + B[thread_idx];
}

template<
    typename FPTYPE,
    int      THREADS_PER_BLOCK> 
__global__ void matrix_add_gpu_rocm_kernel(
    const int row,
    const int col,
    const FPTYPE * A,
    const FPTYPE * B,
    FPTYPE * C)
{
  const int block_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;

  for (int ii = thread_idx; ii < col; ii += THREADS_PER_BLOCK) {
    C[block_idx * col + ii] = A[block_idx * col + ii] + B[block_idx * col + ii];
  }
}

template<typename FPTYPE> 
void vector_add_gpu_rocm(
    const int size,
    const FPTYPE * A,
    const FPTYPE * B,
    FPTYPE * C)
{
  const int nblock = (size + TPB - 1) / TPB;
  vector_add_gpu_rocm_kernel<<<nblock, TPB>>> (
    size, A, B,
    C);
  hipLaunchKernelGGL(
    HIP_KERNEL_NAME(vector_add_gpu_rocm_kernel<FPTYPE, TPB>), 
    nblock, TPB, 0, 0,
    size, A, B,
    C);
  hipErrCheck(hipGetLastError());
  hipErrCheck(hipDeviceSynchronize());
}

template<typename FPTYPE> 
void matrix_add_gpu_rocm(
    const int row,
    const int col,
    const FPTYPE * A,
    const FPTYPE * B,
    FPTYPE * C)
{
  const int nblock = row;
  hipLaunchKernelGGL(
    HIP_KERNEL_NAME(matrix_add_gpu_rocm_kernel<FPTYPE, TPB>), 
    nblock, TPB, 0, 0,
    row, col, A, B,
    C);
  hipErrCheck(hipGetLastError());
  hipErrCheck(hipDeviceSynchronize());
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

template<typename int> void vector_add_gpu_rocm(const int size, const int * A, const int * B, int * C);
template<typename float> void vector_add_gpu_rocm(const int size, const float * A, const float * B, float * C);
template<typename double> void vector_add_gpu_rocm(const int size, const double * A, const double * B, double * C);

template<typename int> void matrix_add_gpu_rocm(const int row, const int col, const int * A, const int * B, int * C);
template<typename float> void matrix_add_gpu_rocm(const int row, const int col, const float * A, const float * B, float * C);
template<typename double> void matrix_add_gpu_rocm(const int row, const int col, const double * A, const double * B, double * C);