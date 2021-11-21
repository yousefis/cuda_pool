/*
 * Developer: Sahar Yousefi
 * more info: https://github.com/yousefis/cuda_pool
 */

#include"../include/kernels.cuh"
#include<iostream>
__global__
void sobel_filter(uchar* img1, uchar* result, size_t height, size_t width)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

    if (row>=width) return;
    if (col>=height) return;
    float Gx = img1[(row-1)*height+(col-1)] - img1[(row-1)*height+(col+1)]+
         2 * img1[row*height+(col-1)] - 2 * img1[row*height+(col+1)]+
         img1[(row+1)*height+(col-1)] - img1[(row+1)*height+(col+1)];
    float Gy = img1[(row-1)*height+(col-1)] + 2 * img1[(row-1)*height+col] + img1[(row-1)*height+(col+1)]+
               (-1) * img1[(row+1)*height+(col-1)] + (-2) * img1[(row+1)*height+col] + (-1) * img1[(row+1)*height+(col+1)];
    result[row*height + col] = sqrt (Gx*Gx + Gy*Gy);
        
}