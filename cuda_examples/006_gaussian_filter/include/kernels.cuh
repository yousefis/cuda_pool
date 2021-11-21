/*
 * Developer: Sahar Yousefi
 * more info: https://github.com/yousefis/cuda_pool
 */
#ifndef KERNELS_CUH
#include <cuda.h>
#include <cuda_runtime_api.h>
#include<stdio.h>
#include<iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
__device__ 
const int DIM=16;
__device__ 
const float gaussian_kernel[3][3] = {
        {1,2,1},
        {2,4,2},
        {1,2,1}
    };
__global__
void gaussian_filter(uchar* img1, uchar* result, size_t height, size_t width);
#endif