/*
 * Developer: Sahar Yousefi
 * more info: https://github.com/yousefis/cuda_pool
 */

#include"../include/kernels.cuh"
#include<iostream>
__global__
void multiply_matrices(float* a, float* b, float* c,  size_t M, size_t N, size_t K)
{
    int col = threadIdx.x + blockIdx.x *blockDim.x;  
    int row = threadIdx.y + blockIdx.y *blockDim.y;

    // c[row, col] += a[row , k] * b[k, col]
    // a:M*N, b:N*K, c: M*K
    if(row>=M) return;
    if(col>=K) return;

    for(int k=0; k<N ;k++)
        c[row*M + col] += a[row*M + k] * b[k*N + col];
}

