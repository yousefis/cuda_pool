/*
 * Developer: Sahar Yousefi
 * more info: https://github.com/yousefis/cuda_pool
 */

#include"../include/kernels.cuh"
#include<iostream>
__global__
void sum_vectors(float* a, float* b, float* c, size_t n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x; //index of the thread on the block and grid
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];       
    }
    
}

