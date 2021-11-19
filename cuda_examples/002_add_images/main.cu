

/*
 * Developer: Sahar Yousefi
 * more info: https://github.com/yousefis/cuda_pool
 */


#include"include/kernels.cuh"

//this file adds two vectors using a cuda kernel
int main(int argc,char* argv[])
{
   assert(argc==4&&"Call: ./main image1 image2 image_result");
  
   cv::Mat img1, img2;
   //Here I have rgb images and just read a channel of the images to make 2D matrices
   cv::extractChannel(cv::imread(argv[1], cv::COLOR_BGR2GRAY),img1,0); 
   cv::extractChannel(cv::imread(argv[2], cv::COLOR_BGR2GRAY),img2,0); 
   
   //convert cv::Mat to an array of uchar
   uchar * uImg1 = img1.isContinuous()? img1.data: img1.clone().data;

   //convert cv::Mat to an array of uchar
   uchar * uImg2 = img2.isContinuous()? img2.data: img2.clone().data;

   //size is the same for both arrays
   size_t length = img1.total()*img1.channels();


   //now we have two arrays of uchar, summing together using cuda and return an array of float
   // define device variables

   uchar *device_img1, *device_img2, *device_sum;
   cudaError res = cudaMalloc((void**)&device_img1, length*sizeof(uchar));
   if (res!=cudaSuccess){
       std::cout<<"Error in memory allocation, file: "<<__FILE__<<", line: "<<__LINE__<<"\n";
   }

   res = cudaMalloc((void**)&device_img2,length*sizeof(uchar));
   if (res!=cudaSuccess){
       std::cout<<"Error in memory allocation, file: "<<__FILE__<<", line: "<<__LINE__<<"\n";
   }

   res = cudaMalloc((void**)& device_sum, length*sizeof(uchar));
   if (res!=cudaSuccess){
       std::cout<<"Error in memory allocation, file: "<<__FILE__<<", line: "<<__LINE__<<"\n";
   }

   //copy memor from host to device
   cudaMemcpy((void*)device_img1, uImg1, length*sizeof(uchar), cudaMemcpyHostToDevice);
   cudaMemcpy((void*)device_img2, uImg2, length*sizeof(uchar), cudaMemcpyHostToDevice);

   dim3 gridDim((img1.size[0]-1)/16 + 1, (img1.size[1]-1)/16 + 1, 1);
   dim3 blockDim(16,16,1);
   printf("w: %d,h: %d\n", img1.size[1], img1.size[0]);
   sum_images<<<gridDim, blockDim>>>(device_img1, device_img2, device_sum, img1.size[1], img1.size[0]);
   


   //copy the result from device to host
   uchar* result = (uchar*) malloc(sizeof(uchar)*length);
   cudaMemcpy((void*)result, device_sum, sizeof(uchar)*length,cudaMemcpyDeviceToHost);

   //convert the result array to cv::Mat and save 
   cv::Mat mat_result(img1.size[0],img1.size[1],CV_8UC1,result);
   cv::imwrite(argv[3], mat_result);
   std::cout<<"The result can be found: "<<argv[3]<<"\n";
   
   //free allocated memory
   cudaFree(device_img1);
   cudaFree(device_img2);
   cudaFree(device_sum);



   //printf("%f\n",float(arr[100,100]));


  


}