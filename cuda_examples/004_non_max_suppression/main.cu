

/*
 * Developer: Sahar Yousefi
 * more info: https://github.com/yousefis/cuda_pool
 */

#include"include/kernels.cuh"
#include<stdio.h>
#include<iostream>

// NMS for yolo output
// the output is an array, the first element is the no of bounding boxes, 
// the rest is the bounding boxes with this structure: [x_c, y_c, w, h, conf, class_id]
// call it like: non_max_suppresion(float* yolo_output, float nms_thresh, float conf)

int main(int argc,char* argv[])
{

}