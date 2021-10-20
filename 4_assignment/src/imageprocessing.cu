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
void callcontrast(dim3 blocks, dim3 threads, unsigned char* out_image, float contrastfactor, unsigned char* d_input, int height, int width){
	contrast <<<blocks, threads>>> (out_image, contrastfactor, d_input, height, width);
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
void callsaturation(dim3 blocks, dim3 threads, unsigned char* out_image, float saturationfactor, unsigned char* d_input, int height, int width){
	saturation <<<blocks, threads>>> (out_image, saturationfactor, d_input, height, width);
}

__global__
void smooth(unsigned char* out_image, unsigned char* in_image, float *conv_kernel, int halfl, int height, int width)
{
	int pos_x = blockIdx.x * blockDim.x + threadIdx.x;//x coordinate of pixel
	int pos_y = blockIdx.y * blockDim.y + threadIdx.y;//y coordinate of pixel

	if (pos_x < width && pos_y < height)
	{
		int l = 2*halfl + 1;
		float size = l*l;
		float r = float(0.0f);
		float g = float(0.0f);
		float b = float(0.0f);
		float originr = ((float)in_image[pos_x * width + pos_y]) / 255.0f;
		float origing = ((float)in_image[(height + pos_x) * width + pos_y]) / 255.0f;
		float originb = ((float)in_image[(height * 2 + pos_x) * width + pos_y]) / 255.0f;
		for(int i=(-halfl); i<=halfl; i++){
			for(int j=(-halfl); j<=halfl; j++){
				int convidx = (i+halfl)*l+j+halfl;
				if(pos_x + i > 0 && pos_y + i > 0 && pos_x + j <= width && pos_y + i <= height)
				{
					r += conv_kernel[convidx]*((float)in_image[(pos_x+i) * width + (pos_y+j)])/255.0f; 
					g += conv_kernel[convidx]*((float)in_image[(height + (pos_x+i)) * width + (pos_y+j)])/255.0f;
					b += conv_kernel[convidx]*((float)in_image[(height * 2 + (pos_x+i)) * width + (pos_y+j)])/255.0f;
				}
			}
		}
		for (int i = (-1 * 15); i <= 15; i++)
			for (int j = (-1 *15); j <= 15; j++)
			{
				if (pos_x + j > 0 && pos_y + i > 0 && pos_x + j <= width && pos_y + i <= height)
				{
					sumR += (float)in_image[(pos_y + i) * width + (pos_x + j)] / (31 * 31);
					sumG += (float)in_image[(height + (pos_y + i)) * width + (pos_x + j)] / (31 * 31);
					sumB += (float)in_image[(height * 2 + (pos_y + i)) * width + (pos_x + j)] / (31 * 31);
				}	
			}
		//sumR = sumR / (15 * 15);
		//sumG = sumG / (15 * 15);
		//sumB = sumB / (15 * 15);
		if (sumR > 255)
			sumR = 255;
		if (sumG > 255)
			sumG = 255;
		if (sumB > 255)
			sumB = 255;
		
		out_image[pos_y * width + pos_x] = (unsigned char)(sumR );
		out_image[(height + pos_y) * width + pos_x] = (unsigned char)(sumG ) ;
		out_image[(height * 2 + pos_y) * width + pos_x] = (unsigned char)(sumB );
		
	}
}
void callsmooth(dim3 blocks, dim3 threads, unsigned char* out_image,  unsigned char* d_input, float *conv_kernel, int length,int height, int width)
{
	int halfl = length/2;
	smooth <<<blocks, threads>>> (out_image, d_input, conv_kernel, halfl, height, width);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
	  printf("Error: %s\n", cudaGetErrorString(err));
}
