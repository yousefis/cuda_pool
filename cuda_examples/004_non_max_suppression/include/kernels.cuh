/*
 * Developer: Sahar Yousefi
 * more info: https://github.com/yousefis/cuda_pool
 */
#ifndef KERNELS_CUH
#include <cuda.h>
#include <cuda_runtime_api.h>
typedef struct bounding_box
{
    int x1,y1,x2,y2;
};

__device__
float Intersection_of_union(bounding_box b1, bounding_box b2);
__global__
void non_max_suppresion(float* yolo_output, float nms_thresh, float conf);
#endif