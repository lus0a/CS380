#include "imageprocessing.cuh"
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <sstream>

#include <device_launch_parameters.h>
// TODO: read about the CUDA programming model: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model
// If everything is setup correctly, this file is compiled by the CUDA/C++ compiler (that is different from the C++ compiler).
// The CUDA/C++ compiler understands certain things that your C++ compiler doesn't understand - like '__global__', 'threadIdx', and function calls with triple-angle brackets, e.g., testArray<<<...>>>();


// do not use this method for anything else than verifying cuda compiled, linked and executed
__global__ void testArray(float* dst, float value) {
	unsigned int index = threadIdx.x;
	dst[index] = value;
}

void testCudaCall() {
	// quick and dirty test of CUDA setup
	const unsigned int N = 1024;
	float* device_array;
	cudaMalloc(&device_array, N * sizeof(float));
	testArray << <1, N >> > (device_array, -0.5f);
	float x[N];
	cudaMemcpy(x, device_array, N * sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "quick and dirty test of CUDA setup: " << x[0] << " " << x[1] << " " << x[1023] << std::endl;
	cudaFree(device_array);
}



// TODO: implement the image processing operations using CUDA kernels
__global__ 
void brightness(unsigned char* out_image, float brightfactor, unsigned char* in_image, int height, int width)
{
	int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
	int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// if(pos_x >= width || pos_y >= height) return;
	if (pos_x < width && pos_y < height)
	{
	unsigned char r = in_image[pos_x * width + pos_y];
	unsigned char g = in_image[(height + pos_x) * width + pos_y];
	unsigned char b = in_image[(height * 2 + pos_x) * width + pos_y];
	
	out_image[pos_x * width + pos_y] = r*brightfactor;
	if(out_image[pos_x * width + pos_y] > 255)
		out_image[pos_x * width + pos_y] = 255;

	out_image[(height + pos_x) * width + pos_y] = g*brightfactor;
	if(out_image[(height + pos_x) * width + pos_y] > 255)
		out_image[(height + pos_x) * width + pos_y] = 255;

	out_image[(height * 2 + pos_x) * width + pos_y] = b*brightfactor;
	if(out_image[(height * 2 + pos_x) * width + pos_y] > 255)
		out_image[(height * 2 + pos_x) * width + pos_y] = 255;
	}
}
void callbrightness(dim3 blocks, dim3 threads, unsigned char* out_image, float brightfactor, unsigned char* d_input, int height, int width){
	brightness <<<blocks, threads>>> (out_image, brightfactor, d_input, height, width);
}

__global__ 
void contrast(unsigned char* out_image, float contrastfactor, unsigned char* in_image, int height, int width){
  	int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
	int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
	//printf("pos_x and pos_y is %d %d", pos_x, pos_y);
	if(pos_x >= width || pos_y >= height) return;
	
	unsigned char r = in_image[pos_x * width + pos_y];
  	unsigned char g = in_image[(height + pos_x) * width + pos_y];
  	unsigned char b = in_image[(height * 2 + pos_x) * width + pos_y];
  
	out_image[pos_x * width + pos_y] = r*contrastfactor + (1-contrastfactor)*255;
	if(out_image[pos_x * width + pos_y] > 255)
		out_image[pos_x * width + pos_y] = 255;

	out_image[(height + pos_x) * width + pos_y] = g*contrastfactor + (1-contrastfactor)*255;
	if(out_image[(height + pos_x) * width + pos_y] > 255)
	  	out_image[(height + pos_x) * width + pos_y] = 255;

	out_image[(height * 2 + pos_x) * width + pos_y] = b*contrastfactor + (1-contrastfactor)*255;
	if(out_image[(height * 2 + pos_x) * width + pos_y]>255)
		out_image[(height * 2 + pos_x) * width + pos_y] = 255;
}
void callcontrast(dim3 grid, dim3 block, unsigned char* out_image, float contrastfactor, unsigned char* d_input, int height, int width){
	contrast <<<grid, block>>> (out_image, contrastfactor, d_input, height, width);
}

__global__
void saturation(unsigned char* out_image, float saturationfactor, unsigned char* in_image, int height, int width){
  	int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
	int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
	//printf("pos_x and pos_y is %d %d", pos_x, pos_y);
	if(pos_x >= width || pos_y >= height) return;
	
	unsigned char r = in_image[pos_x * width + pos_y];
  	unsigned char g = in_image[(height + pos_x) * width + pos_y];
  	unsigned char b = in_image[(height * 2 + pos_x) * width + pos_y];
	float intensity = (54.1875*r+182.427*g+18.3855*b) / 255.0;
	out_image[pos_x * width + pos_y] = r*saturationfactor + (1-saturationfactor)*intensity;
	if(out_image[pos_x * width + pos_y] > 255)
		out_image[pos_x * width + pos_y] = 255;

	out_image[(height + pos_x) * width + pos_y] = g*saturationfactor + (1-saturationfactor)*intensity;
	if(out_image[(height + pos_x) * width + pos_y] > 255)
	  out_image[(height + pos_x) * width + pos_y] = 255;
	  
	out_image[(height * 2 + pos_x) * width + pos_y] = b*saturationfactor + (1-saturationfactor)*intensity;
	if(out_image[(height * 2 + pos_x) * width + pos_y]>255)
		out_image[(height * 2 + pos_x) * width + pos_y] = 255;
}
void callsaturation(dim3 grid, dim3 block, unsigned char* out_image, float saturationfactor, unsigned char* d_input, int height, int width){
	saturation <<<grid, block>>> (out_image, saturationfactor, d_input, height, width);
}