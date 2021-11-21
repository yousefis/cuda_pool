

/*
 * Developer: Sahar Yousefi
 * more info: https://github.com/yousefis/cuda_pool
 */

#include"include/kernels.cuh"
#include<stdio.h>
#include<iostream>


void fill_matrix(float* matrix, int w, int h)
{
    std::cout<<w<<","<<h<<"\n";
    int p=0;
    for (int r=0; r<w; r++)
        for(int c=0;c<h;c++)        
            matrix[r*h+c] = p++;// r*w+c;
            

}
void print_matrix(float* array, int W, int H)
{
    std::cout<<"\n------\n";
    for (int r=0; r<W; r++)
    {
        for (int c=0; c<H; c++)
        {
            std::cout<<array[r*H+c]<<" ";
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
    const int M=4, N=3, K=5;
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
    const clock_t begin_time = clock();

    dim3 gridDim((M-1)/16+1, (K-1)/16+1, 1);
    dim3 blockDim(16,16,1);
    std::cout<<"Multiplication of two matrices with the below sizes:\n";
    std::cout<<"A_("<<M<<"*"<<N<<") * B_("<<N<<"*"<<K<<") = C_("<<M<<"*"<<K<<")\n\n";
    multiply_matrices<<<gridDim, blockDim>>>(device_a, device_b, device_c, M,N,K);
    std::cout <<"Elapsed time: "<< float( clock () - begin_time ) /  CLOCKS_PER_SEC<<"\n\n";

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