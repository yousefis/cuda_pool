

/*
 * Developer: Sahar Yousefi
 * more info: https://github.com/yousefis/cuda_pool
 */

#include"include/kernels.cuh"
#include<stdio.h>
#include<iostream>

//this file adds two vectors using a cuda kernel
int main(int argc,char* argv[])
{
    float *host_a, *host_b, *host_c; // variables on cpu
    float *device_a, *device_b, *device_c; //varibales on gpu
    

    //######################################################
    //Allocating memory for the variables on host
    int LENGTH = 100000; 
    size_t size = sizeof(float)*LENGTH;
    host_a = (float*) malloc(size);
    host_b = (float*) malloc(size);
    host_c = (float*) malloc(size);

    //######################################################
    //Initialize host_a, host_b
    for (int i=0; i<LENGTH; i++)
    {
        host_a[i] = i*1.;
        host_b[i] = i*1.;
    }


    //######################################################
    //Allocating memory for the variables on device
    cudaError res;
    res = cudaMalloc((void**)&device_a, size);
    if (res !=cudaSuccess)
    {
        printf("Error in A mem allocation %s, in file: %s, in line: %s", res, __FILE__, __LINE__);
    }
    

    res = cudaMalloc((void**)&device_b, size);
    if (res !=cudaSuccess)
    {
        printf("Error in B mem allocation %s, in file: %s, in line: %s", res, __FILE__, __LINE__);
    }  

    res = cudaMalloc((void**)&device_c, size);
    if (res !=cudaSuccess)
    {
        printf("Error in B mem allocation %s, in file: %s, in line: %s", res, __FILE__, __LINE__);
    }  
    //###################################################### 
    //copy cpu variables to gpu variables
    cudaMemcpy((void*)device_a, host_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)device_b, host_b, size, cudaMemcpyHostToDevice);

    //######################################################
    // calling the kernel
    dim3 dimGrid((LENGTH-1/255)+1 , 1, 1);
    dim3 dimBlock(256,1,1);
    sum_vectors <<<dimGrid,dimBlock>>> (device_a, device_b, device_c, LENGTH);

    //######################################################
    //copy the result to the host
    cudaMemcpy((void*)host_c, device_c, size, cudaMemcpyDeviceToHost);

    //######################################################
    //uncomment to print out the result
    // for(int i=0; i<LENGTH; i++)
    // {
    //     printf("%f, ",host_c[i]);
    // }

    //######################################################
    // Free the allocated memory on device
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    //######################################################
    // Free the allocated memory on host
    free(host_a);
    free(host_b);
    free(host_c);
    


    // cudaMalloc()
}