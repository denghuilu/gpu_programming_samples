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
#include<hip/hip_runtime.h>

#define hipErrCheck(res) {hipAssert((res), __FILE__, __LINE__);}
inline void hipAssert(hipError_t code, const char *file, int line, bool abort=true) {
  if (code != hipSuccess) {
    fprintf(stderr,"hip assert: %s %s %d\n", hipGetErrorString(code), file, line);
  }
}
  
inline void getDeviceCount(int &gpu_num) { hipGetDeviceCount(&gpu_num) ;}

template <typename FPTYPE>
void memcpy_host_to_device(
    FPTYPE * device, 
    const std::vector<FPTYPE> &host) 
{
  hipErrCheck(hipMemcpy(device, &host[0], sizeof(FPTYPE) * host.size(), hipMemcpyHostToDevice));  
}

template <typename FPTYPE>
void memcpy_host_to_device(
    FPTYPE * device, 
    const FPTYPE * host,
    const int size) 
{
  hipErrCheck(hipMemcpy(device, host, sizeof(FPTYPE) * size, hipMemcpyHostToDevice));  
}

template <typename FPTYPE>
void memcpy_device_to_host(
    const FPTYPE * device, 
    std::vector<FPTYPE> &host) 
{
  hipErrCheck(hipMemcpy(&host[0], device, sizeof(FPTYPE) * host.size(), hipMemcpyDeviceToHost));  
}

template <typename FPTYPE>
void memcpy_device_to_host(
    const FPTYPE * device, 
    FPTYPE * host,
    const int size) 
{
  hipErrCheck(hipMemcpy(host, device, sizeof(FPTYPE) * size, hipMemcpyDeviceToHost));  
}

template <typename FPTYPE>
void malloc_device_memory(
    FPTYPE * &device, 
    const std::vector<FPTYPE> &host) 
{
  hipErrCheck(hipMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
}

template <typename FPTYPE>
void malloc_device_memory(
    FPTYPE * &device, 
    const int size) 
{
  hipErrCheck(hipMalloc((void **)&device, sizeof(FPTYPE) * size));
}

template <typename FPTYPE>
void malloc_device_memory_sync(
    FPTYPE * &device,
    const std::vector<FPTYPE> &host) 
{
  hipErrCheck(hipMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
  memcpy_host_to_device(device, host);
}

template <typename FPTYPE>
void malloc_device_memory_sync(
    FPTYPE * &device,
    const FPTYPE * host,
    const int size)
{
  hipErrCheck(hipMalloc((void **)&device, sizeof(FPTYPE) * size));
  memcpy_host_to_device(device, host, size);
}

template <typename FPTYPE>
void delete_device_memory(
    FPTYPE * &device) 
{
  if (device != NULL) {
    hipErrCheck(hipFree(device));
  }
}

template <typename FPTYPE>
void memset_device_memory(
    FPTYPE * device, 
    const int var,
    const int size) 
{
  hipErrCheck(hipMemset(device, var, sizeof(FPTYPE) * size));  
}
} // end of namespace deepmd