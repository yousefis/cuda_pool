/*
 * Developer: Sahar Yousefi
 * more info: https://github.com/yousefis/cuda_pool
 */
#ifndef KERNELS_CUH
#include <cuda.h>
#include <cuda_runtime_api.h>
__global__
void multiply_matrices(float* a, float* b, float* c, size_t M, size_t N, size_t K);
#endif