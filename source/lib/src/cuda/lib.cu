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

template<
    typename FPTYPE,
    int      THREADS_PER_BLOCK> 
__global__ void vector_add_gpu_cuda_kernel(
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
__global__ void matrix_add_gpu_cuda_kernel(
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
void vector_add_gpu_cuda(
    const int size,
    const FPTYPE * A,
    const FPTYPE * B,
    FPTYPE * C)
{ 
  const int nblock = (size + TPB - 1) / TPB;
  vector_add_gpu_cuda_kernel<FPTYPE, TPB> <<<nblock, TPB>>> (
    size, A, B,
    C);
  cudaErrCheck(cudaGetLastError());
  cudaErrCheck(cudaDeviceSynchronize());
}

template<typename FPTYPE> 
void matrix_add_gpu_cuda(
    const int row,
    const int col,
    const FPTYPE * A,
    const FPTYPE * B,
    FPTYPE * C)
{
  const int nblock = row;
  matrix_add_gpu_cuda_kernel<FPTYPE, TPB> <<<nblock, TPB>>> (
    row, col, A, B,
    C);
  cudaErrCheck(cudaGetLastError());
  cudaErrCheck(cudaDeviceSynchronize());
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

template void vector_add_gpu_cuda<int>(const int size, const int * A, const int * B, int * C);
template void vector_add_gpu_cuda<float>(const int size, const float * A, const float * B, float * C);
template void vector_add_gpu_cuda<double>(const int size, const double * A, const double * B, double * C);

template void matrix_add_gpu_cuda<int>(const int row, const int col, const int * A, const int * B, int * C);
template void matrix_add_gpu_cuda<float>(const int row, const int col, const float * A, const float * B, float * C);
template void matrix_add_gpu_cuda<double>(const int row, const int col, const double * A, const double * B, double * C);
