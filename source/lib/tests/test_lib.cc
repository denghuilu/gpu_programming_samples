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

#include <math.h>
#include <iostream>
#include <gtest/gtest.h>
#include "lib.h"
#include "device.h"

class TestLib : public ::testing::Test
{
protected:
  std::vector<double > A = {
    1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0
  };
  std::vector<double > B = {
    2.0, 2.0, 2.0, 2.0,
		2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0
  };
  std::vector<double > C = {
    0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0
  };
  
  std::vector<double > expected_C = {
    3.0, 3.0, 3.0, 3.0,
		3.0, 3.0, 3.0, 3.0,
    3.0, 3.0, 3.0, 3.0,
    3.0, 3.0, 3.0, 3.0 
  };  

  const int length = 16;
  const int row = 4;
  const int col = 4;
  
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TestLib, vector_add_cpu) {
  vector_add_cpu(length, &A[0], &B[0], &C[0]);

  for (int ii = 0; ii < length; ii++) {
  	EXPECT_LT(fabs(C[ii] - expected_C[ii]) , 1e-5);
  }
}

TEST_F(TestLib, matrix_add_cpu) {
  matrix_add_cpu(row, col, &A[0], &B[0], &C[0]);

  for (int ii = 0; ii < row; ii++) {
    for (int jj = 0; jj < col; jj++) {
  	  EXPECT_LT(fabs(C[ii * col + jj] - expected_C[ii * col + jj]) , 1e-5);
    }
  }
}

#if USE_CUDA_TOOLKIT
TEST_F(TestLib, vector_add_gpu_cuda) {
  double * d_A = NULL, * d_B = NULL, * d_C = NULL;
  malloc_device_memory_sync(d_A, A);
  malloc_device_memory_sync(d_B, B);
  malloc_device_memory_sync(d_C, C);
  vector_add_gpu_cuda(length, d_A, d_B, d_C);
  memcpy_device_to_host(d_C, C);
  delete_device_memory(d_A);
  delete_device_memory(d_B);
  delete_device_memory(d_C);

  for (int ii = 0; ii < length; ii++) {
  	EXPECT_LT(fabs(C[ii] - expected_C[ii]) , 1e-5);
  }
}

TEST_F(TestLib, matrix_add_gpu_cuda) {
  double * d_A = NULL, * d_B = NULL, * d_C = NULL;
  malloc_device_memory_sync(d_A, A);
  malloc_device_memory_sync(d_B, B);
  malloc_device_memory_sync(d_C, C);
  matrix_add_gpu_cuda(row, col, d_A, d_B, d_C);
  memcpy_device_to_host(d_C, C);
  delete_device_memory(d_A);
  delete_device_memory(d_B);
  delete_device_memory(d_C);

  for (int ii = 0; ii < row; ii++) {
    for (int jj = 0; jj < col; jj++) {
  	  EXPECT_LT(fabs(C[ii * col + jj] - expected_C[ii * col + jj]) , 1e-5);
    }
  }
}
#endif //USE_CUDA_TOOLKIT

#if USE_ROCM_TOOLKIT
TEST_F(TestLib, vector_add_gpu_rocm) {
  double * d_A = NULL, * d_B = NULL, * d_C = NULL;
  malloc_device_memory_sync(d_A, A);
  malloc_device_memory_sync(d_B, B);
  malloc_device_memory_sync(d_C, C);
  vector_add_gpu_rocm(length, d_A, d_B, d_C);
  memcpy_device_to_host(d_C, C);
  delete_device_memory(d_A);
  delete_device_memory(d_B);
  delete_device_memory(d_C);

  for (int ii = 0; ii < length; ii++) {
  	EXPECT_LT(fabs(C[ii] - expected_C[ii]) , 1e-5);
  }
}

TEST_F(TestLib, matrix_add_gpu_rocm) {
  double * d_A = NULL, * d_B = NULL, * d_C = NULL;
  malloc_device_memory_sync(d_A, A);
  malloc_device_memory_sync(d_B, B);
  malloc_device_memory_sync(d_C, C);
  matrix_add_gpu_rocm(row, col, d_A, d_B, d_C);
  memcpy_device_to_host(d_C, C);
  delete_device_memory(d_A);
  delete_device_memory(d_B);
  delete_device_memory(d_C);

  for (int ii = 0; ii < row; ii++) {
    for (int jj = 0; jj < col; jj++) {
  	  EXPECT_LT(fabs(C[ii * col + jj] - expected_C[ii * col + jj]) , 1e-5);
    }
  }
}
#endif //USE_ROCM_TOOLKIT
