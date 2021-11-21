/*
 * Developer: Sahar Yousefi
 * more info: https://github.com/yousefis/cuda_pool
 */

#include"../include/kernels.cuh"
#include<iostream>
__device__
float Intersection_of_union(bounding_box bb1, bounding_box bb2)
{

    float area_bb1 =  (float)(bb1.x2 - bb1.x1)*(bb1.y2 - bb1.y1);
    float area_bb2 = (float)(bb2.x2 - bb2.x1)*(bb2.y2 - bb2.y1);
    float x1_max, x2_min, y1_max, y2_min;

    x1_max = max(bb1.x1,bb2.x1);
    y1_max = max(bb1.y1,bb2.y1);
    x2_min = min((bb1.x2),(bb2.x2));
    y2_min = min((bb1.y2),(bb2.y2));

    float w = (float)max((float)0, x2_min - x1_max);  
    float h = (float)max((float)0, y2_min - y1_max);  
    float inter = ((w*h)/(area_bb1 + area_bb2 - w*h));
    return inter;
}
__global__
void non_max_suppresion(float* yolo_output, float nms_thresh, float conf)
{
	int x =(blockIdx.x * blockDim.x + threadIdx.x);
	int y =(blockIdx.y * blockDim.y + threadIdx.y);

	int num_boxes = int(*yolo_output);
        if (x >= *yolo_output|| y >= *yolo_output)
		return;
	if (x==y)
		return;
	int offset = 1; //offset for the begining of the bboxes in the matrix
    float bb_conf = yolo_output[x * 6 + 4 + offset];
    
	if (bb_conf < conf) { //just marks as it should be removed
		yolo_output[x * 6 + 4 + offset] = 0; //confidence
		return; 
	}

	bounding_box box1 = {yolo_output[x * 6 + offset] - yolo_output[x * 6+2 + offset]/2, 
		    yolo_output[x * 6 + 1 + offset] - yolo_output[x * 6+3 + offset]/2,
		    yolo_output[x * 6 + offset] + yolo_output[x * 6+2 + offset]/2, 
		    yolo_output[x * 6 + 1 + offset] + yolo_output[x * 6+3 + offset]/2};

	bounding_box box2 = {yolo_output[y * 6 + offset] - yolo_output[y * 6+2 + offset]/2, 
		    yolo_output[y * 6 + 1 +offset] - yolo_output[y * 6+3 + offset]/2, 
		    yolo_output[y * 6 + offset] + yolo_output[y * 6+2 + offset]/2, 
		    yolo_output[y * 6 + 1 +offset] + yolo_output[y * 6+3 + offset]/2};


	if (Intersection_of_union(box1, box2) > nms_thresh )
		if ( (yolo_output [y * 6 + 5 + offset] == yolo_output[ x * 6 + 5 + offset]) &&
		     (yolo_output[ y * 6 + 4 + offset] <= bb_conf) ) 
		{
			yolo_output[ y * 6 + 4 + offset] = 0; //confidence
		}
}