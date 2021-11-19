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
__global__
void sum_images(uchar* img1, uchar* img2, uchar* result, size_t height, size_t width);
#endif