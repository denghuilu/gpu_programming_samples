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

#pragma once
#include <vector>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define cudaErrCheck(res) {cudaAssert((res), __FILE__, __LINE__);}
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"cuda assert: %s %s %d\n", cudaGetErrorString(code), file, line);
  }
}
  
inline int get_device_count() {
  int gpu_num = 0; 
  cudaErrCheck(cudaGetDeviceCount(&gpu_num));
  return gpu_num;
}

template <typename FPTYPE>
void memcpy_host_to_device(
    FPTYPE * device, 
    const std::vector<FPTYPE> &host) 
{
  cudaErrCheck(cudaMemcpy(device, &host[0], sizeof(FPTYPE) * host.size(), cudaMemcpyHostToDevice));  
}

template <typename FPTYPE>
void memcpy_host_to_device(
    FPTYPE * device, 
    const FPTYPE * host,
    const int size) 
{
  cudaErrCheck(cudaMemcpy(device, host, sizeof(FPTYPE) * size, cudaMemcpyHostToDevice));  
}

template <typename FPTYPE>
void memcpy_device_to_host(
    const FPTYPE * device, 
    std::vector<FPTYPE> &host) 
{
  cudaErrCheck(cudaMemcpy(&host[0], device, sizeof(FPTYPE) * host.size(), cudaMemcpyDeviceToHost));  
}

template <typename FPTYPE>
void memcpy_device_to_host(
    const FPTYPE * device, 
    FPTYPE * host,
    const int size) 
{
  cudaErrCheck(cudaMemcpy(host, device, sizeof(FPTYPE) * size, cudaMemcpyDeviceToHost));  
}

template <typename FPTYPE>
void malloc_device_memory(
    FPTYPE * &device, 
    const std::vector<FPTYPE> &host) 
{
  cudaErrCheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
}

template <typename FPTYPE>
void malloc_device_memory(
    FPTYPE * &device, 
    const int size) 
{
  cudaErrCheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * size));
}

template <typename FPTYPE>
void malloc_device_memory_sync(
    FPTYPE * &device,
    const std::vector<FPTYPE> &host) 
{
  cudaErrCheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
  memcpy_host_to_device(device, host);
}

template <typename FPTYPE>
void malloc_device_memory_sync(
    FPTYPE * &device,
    const FPTYPE * host,
    const int size)
{
  cudaErrCheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * size));
  memcpy_host_to_device(device, host, size);
}

template <typename FPTYPE>
void delete_device_memory(
    FPTYPE * &device) 
{
  if (device != NULL) {
    cudaErrCheck(cudaFree(device));
  }
}

template <typename FPTYPE>
void memset_device_memory(
    FPTYPE * device, 
    const int var,
    const int size) 
{
  cudaErrCheck(cudaMemset(device, var, sizeof(FPTYPE) * size));  
}