

/*
 * Developer: Sahar Yousefi
 * more info: https://github.com/yousefis/cuda_pool
 */


#include"include/kernels.cuh"

//this file adds two vectors using a cuda kernel
int main(int argc,char* argv[])
{
   assert(argc==3&&"Call: ./main image image_result");
  
   cv::Mat img1, img2;
   //Here I have rgb images and just read a channel of the images to make 2D matrices
   cv::extractChannel(cv::imread(argv[1], cv::COLOR_BGR2GRAY),img1,0); 
   
   //convert cv::Mat to an array of uchar
   uchar * uImg1 = img1.isContinuous()? img1.data: img1.clone().data;


   //size is the same for both arrays
   size_t length = img1.total()*img1.channels();


   //now we have two arrays of uchar, summing together using cuda and return an array of float
   // define device variables

   uchar *device_img1, *device_sobel;
   cudaError res = cudaMalloc((void**)&device_img1, length*sizeof(uchar));
   if (res!=cudaSuccess){
       std::cout<<"Error in memory allocation, file: "<<__FILE__<<", line: "<<__LINE__<<"\n";
   }


   res = cudaMalloc((void**)& device_sobel, length*sizeof(uchar));
   if (res!=cudaSuccess){
       std::cout<<"Error in memory allocation, file: "<<__FILE__<<", line: "<<__LINE__<<"\n";
   }

   //copy memor from host to device
   cudaMemcpy((void*)device_img1, uImg1, length*sizeof(uchar), cudaMemcpyHostToDevice);

   dim3 gridDim((img1.size[0]-1)/16 + 1, (img1.size[1]-1)/16 + 1, 1);
   dim3 blockDim(16,16,1);
   printf("w: %d,h: %d\n", img1.size[1], img1.size[0]);
   sobel_filter<<<gridDim, blockDim>>>(device_img1, device_sobel, img1.size[1], img1.size[0]);
   


   //copy the result from device to host
   uchar* result = (uchar*) malloc(sizeof(uchar)*length);
   cudaMemcpy((void*)result, device_sobel, sizeof(uchar)*length,cudaMemcpyDeviceToHost);

   //convert the result array to cv::Mat and save 
   cv::Mat mat_result(img1.size[0],img1.size[1],CV_8UC1,result);
   cv::imwrite(argv[2], mat_result);
   std::cout<<"The result can be found: "<<argv[2]<<"\n";
   
   //free allocated memory
   cudaFree(device_img1);
   cudaFree(device_sobel);
   free(result);



}