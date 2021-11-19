/*
 * Developer: Sahar Yousefi
 * more info: https://github.com/yousefis/cuda_pool
 */

#include"../include/kernels.cuh"
#include<iostream>
__global__
void sum_images(uchar* img1, uchar* img2,  uchar* result, size_t height, size_t width)
{
    int c = threadIdx.x + blockIdx.x * blockDim.x;
    int r = threadIdx.y + blockIdx.y * blockDim.y;
    if (c >= width) return;
    if (r >= height) return;

    //sum half of the first image with the 2nd image and use a filter to pass the values below 255
    result[c+r*width] = uchar((int(img1[c+r*width])/2 + int(img2[c+r*width]))>255?255:(int(img1[c+r*width])/2 + int(img2[c+r*width])));
    
}