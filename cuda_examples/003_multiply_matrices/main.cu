

/*
 * Developer: Sahar Yousefi
 * more info: https://github.com/yousefis/cuda_pool
 */

#include"include/kernels.cuh"
#include<stdio.h>
#include<iostream>


void fill_matrix(float* matrix, int w, int h)
{
    for(int c=0;c<w;c++)
        for (int r=0; r<h; r++)
            matrix[r*w+c] = r*w+c;
}
void print_matrix(float* array, int W, int H)
{
    std::cout<<"\n------\n";
    for (int i=0; i<W; i++)
    {
        for (int j=0; j<H; j++)
        {
            std::cout<<array[i*W+j]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n------\n";
}
//this file adds two vectors using a cuda kernel
int main(int argc,char* argv[])
{
    float *host_a, *host_b, *host_c;
    float *device_a, *device_b, *device_c;
    const int M=5, N=3, K=7;
    // initialize the host variables
    host_a = (float*) malloc(M*N*sizeof(float));
    host_b = (float*) malloc(N*K*sizeof(float));
    host_c = (float*) malloc(M*K*sizeof(float));

    fill_matrix(host_a, M, N);
    fill_matrix(host_b, N, K);
    
    // allocate memory for device variables
    cudaMalloc((void**)&device_a, M*N*sizeof(float));
    cudaMalloc((void**)&device_b, N*K*sizeof(float));
    cudaMalloc((void**)&device_c, M*K*sizeof(float));
               

    // copy variables from host to device
    cudaMemcpy((void*)device_a, host_a, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)device_b, host_b, N*K*sizeof(float), cudaMemcpyHostToDevice);

    // call kernel
    dim3 gridDim((M-1)/16+1, (K-1)/16+1, 1);
    dim3 blockDim(16,16,1);
    std::cout<<M<<", "<<N<<", "<<K<<"\n\n";
    multiply_matrices<<<gridDim, blockDim>>>(device_a, device_b, device_c, M,N,K);

    // copy result to host
    cudaMemcpy((void*)host_c , device_c, K*M*sizeof(float), cudaMemcpyDeviceToHost);

    // print out the result
    print_matrix(host_a, M, N);
    print_matrix(host_b, N, K);
    print_matrix(host_c, M,K);
    // free memory
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    
    free(host_a);
    free(host_b);
    free(host_c);

    
}