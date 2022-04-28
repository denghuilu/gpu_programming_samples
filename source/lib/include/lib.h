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
#include <string>
#include <iostream>
#include <stdio.h>
#include <string.h>

template<typename FPTYPE> 
void init_data(
    const int flag,
    const int dim,
    const std::string device,
    FPTYPE * &A,
    FPTYPE * &B,
    FPTYPE * &C);

template<typename FPTYPE> 
void delete_data(
    const int flag,
    const int dim,
    const std::string device,
    FPTYPE * &A,
    FPTYPE * &B,
    FPTYPE * &C);

template<typename FPTYPE> 
void vector_add_cpu(
    const int size,
    const FPTYPE * A,
    const FPTYPE * B,
    FPTYPE * C);

template<typename FPTYPE> 
void matrix_add_cpu(
    const int row,
    const int col,
    const FPTYPE * A,
    const FPTYPE * B,
    FPTYPE * C);


#if USE_CUDA_TOOLKIT
template<typename FPTYPE> 
void vector_add_gpu_cuda(
    const int size,
    const FPTYPE * A,
    const FPTYPE * B,
    FPTYPE * C);

template<typename FPTYPE> 
void matrix_add_gpu_cuda(
    const int row,
    const int col,
    const FPTYPE * A,
    const FPTYPE * B,
    FPTYPE * C);
#endif


#if USE_ROCM_TOOLKIT
template<typename FPTYPE> 
void vector_add_gpu_rocm(
    const int size,
    const FPTYPE * A,
    const FPTYPE * B,
    FPTYPE * C);

template<typename FPTYPE> 
void matrix_add_gpu_rocm(
    const int row,
    const int col,
    const FPTYPE * A,
    const FPTYPE * B,
    FPTYPE * C);
#endif


