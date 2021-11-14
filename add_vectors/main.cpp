#include"include/kernels.cuh"
#include<stdio.h>
#include<iostream>

//this file adds two vectors using a cuda kernel
int main(int argc,char* argv[])
{
    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c;

    //######################################################
    //Allocating memory for the variables on host
    int LENGTH = 100; 
    size_t size = sizeof(float)*LENGTH;
    d_a = (float*) malloc(size);
    d_b = (float*) malloc(size);

    //######################################################
    //Allocating memory for the variables on device
    cudaError res;
    res = cudaMalloc((void**)d_a, size);
    if (res !=cudaSuccess)
    {
        printf("Error in A mem allocation %s, in file: %s, in line: %s", res, __FILE__, __LINE__);
    }
    cudaMemcpy((void*)h_a, d_a, size, cudaMemcpyHostToDevice);

    res = cudaMalloc((void**)d_b, size);
    if (res !=cudaSuccess)
    {
        printf("Error in B mem allocation %s, in file: %s, in line: %s", res, __FILE__, __LINE__);
    }    
    cudaMemcpy((void*)h_b, d_b, size, cudaMemcpyHostToDevice);

    //######################################################
    // Calling the kernel
    //sum_vectors()

    //######################################################
    // Free the allocated memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    // cudaFree(d_c);

    //######################################################
    // Free the allocated memory on host
    // free(h_a);
    // free(h_b);
    // free(h_c);
    


    // cudaMalloc()
}