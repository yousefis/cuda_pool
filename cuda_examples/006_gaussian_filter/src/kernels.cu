/*
 * Developer: Sahar Yousefi
 * more info: https://github.com/yousefis/cuda_pool
 */

#include"../include/kernels.cuh"
#include<iostream>

__global__
void gaussian_filter(uchar* img1, uchar* result, size_t height, size_t width)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

   
    if (row>=width || row <1) return;
    if (col>=height|| col<1) return;

    result[row*height+col] = uchar(min((float(img1[(row-1)*height+(col-1)])*gaussian_kernel[0][0]+float(img1[(row)*height+(col)])*gaussian_kernel[0][1]+float(img1[(row-1)*height+(col+1)])*gaussian_kernel[0][2]+
                             float(img1[(row-1)*height+(col-1)])*gaussian_kernel[1][0]+float(img1[(row)*height+(col)])*gaussian_kernel[1][1]+float(img1[(row-1)*height+(col+1)])*gaussian_kernel[1][2]+
                             float(img1[(row-1)*height+(col-1)])*gaussian_kernel[2][0]+float(img1[(row)*height+(col)])*gaussian_kernel[2][1]+float(img1[(row-1)*height+(col+1)])*gaussian_kernel[2][2])/DIM,255.));
    //printf("%f,",result[row*width+col] );
}