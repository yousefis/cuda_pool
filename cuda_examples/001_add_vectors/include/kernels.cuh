/*
 * Developer: Sahar Yousefi
 * more info: https://github.com/yousefis/cuda_pool
 */
#ifndef KERNELS_CUH
#include <cuda.h>
#include <cuda_runtime_api.h>
__global__
void sum_vectors(float* a, float* b, float* c, size_t n);
#endif