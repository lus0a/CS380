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
	testArray <<<1, N >>> (device_array, -0.5f);
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
					r += conv_kernel[convidx]*((float)in_image[(pos_x+i) * width + (pos_y+j)]); 
					g += conv_kernel[convidx]*((float)in_image[(height + (pos_x+i)) * width + (pos_y+j)]);
					b += conv_kernel[convidx]*((float)in_image[(height * 2 + (pos_x+i)) * width + (pos_y+j)]);
				}
			}
		}
		r /= size;
		g /= size;
		b /= size;
		out_image[pos_x * width + pos_y] = (unsigned char)(r);
		out_image[(height + pos_x) * width + pos_y] = (unsigned char)(g);
		out_image[(height * 2 + pos_x) * width + pos_y] = (unsigned char)(b);
		
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

__global__
void Unrollsmooth(unsigned char* out_image, unsigned char* in_image, float *conv_kernel, int halfl, int height, int width)
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
		#pragma unroll (9)
		for(int i=(-4); i<=4; i++){
			for(int j=(-4); j<=4; j++){
				int convidx = (i+halfl)*l+j+halfl;
				if(pos_x + i > 0 && pos_y + i > 0 && pos_x + j <= width && pos_y + i <= height)
				{
					r += conv_kernel[convidx]*((float)in_image[(pos_x+i) * width + (pos_y+j)]); 
					g += conv_kernel[convidx]*((float)in_image[(height + (pos_x+i)) * width + (pos_y+j)]);
					b += conv_kernel[convidx]*((float)in_image[(height * 2 + (pos_x+i)) * width + (pos_y+j)]);
				}
			}
		}
		r /= size;
		g /= size;
		b /= size;
		out_image[pos_x * width + pos_y] = (unsigned char)(r);
		out_image[(height + pos_x) * width + pos_y] = (unsigned char)(g);
		out_image[(height * 2 + pos_x) * width + pos_y] = (unsigned char)(b);
		
	}
}
void callUnrollsmooth(dim3 blocks, dim3 threads, unsigned char* out_image,  unsigned char* d_input, float *conv_kernel, int length,int height, int width)
{
	int halfl = length/2;
	Unrollsmooth <<<blocks, threads>>> (out_image, d_input, conv_kernel, halfl, height, width);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
	  printf("Error: %s\n", cudaGetErrorString(err));
}


__global__
void edgedetection(unsigned char* out_image, unsigned char* in_image, int height, int width)
{
	int pos_x = blockIdx.x * blockDim.x + threadIdx.x;//pixel coordinate
	int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (pos_x < width - 1 && pos_x >= 1 && pos_y < height - 1 && pos_y >=1)
	{
		//float r = float(0.0f);
		//float g = float(0.0f);
		//float b = float(0.0f);
		float originr = ((float)in_image[pos_x * width + pos_y]);
		float origing = ((float)in_image[(height + pos_x) * width + pos_y]);
		float originb = ((float)in_image[(height * 2 + pos_x) * width + pos_y]);

		float s00 = (float)in_image[(pos_x - 1) * width + (pos_y + 1)];
		float s10 = (float)in_image[(pos_x - 1) * width + (pos_y)];
		float s20 = (float)in_image[(pos_x - 1) * width + (pos_y - 1)];
		float s01 = (float)in_image[(pos_x) * width + (pos_y + 1)];
		float s21 = (float)in_image[(pos_x) * width + (pos_y - 1)];
		float s02 = (float)in_image[(pos_x + 1) * width + (pos_y + 1)];
		float s12 = (float)in_image[(pos_x + 1)*width + (pos_y)];
		float s22 = (float)in_image[(pos_x + 1) * width + (pos_y - 1)];
		float sx = s00 + 2 * s10 + s20 - (s02 + 2 * s12 + s22);
		float sy = s00 + 2 * s01 + s02 - (s20 + 2 * s21 + s22);
		float r = sqrt (sx * sx + sy * sy);
		//r = 0.0f;
		
		s00 = (float)in_image[(height + pos_x - 1) * width + (pos_y + 1)];
		s10 = (float)in_image[(height + pos_x - 1) * width + (pos_y)];
		s20 = (float)in_image[(height + pos_x - 1) * width + (pos_y - 1)];
		s01 = (float)in_image[(height + pos_x)*width + (pos_y + 1)];
		s21 = (float)in_image[(height + pos_x)*width + (pos_y - 1)];
		s02 = (float)in_image[(height + pos_x + 1) * width + (pos_y + 1)];
		s12 = (float)in_image[(height + pos_x + 1) * width + (pos_y)];
		s22 = (float)in_image[(height + pos_x + 1) * width + (pos_y - 1)];
		sx = s00 + 2 * s10 + s20 - (s02 + 2 * s12 + s22);
		sy = s00 + 2 * s01 + s02 - (s20 + 2 * s21 + s22);
		float g = sqrt(sx * sx + sy * sy);
		//g = 0.0f;

		s00 = (float)in_image[(2 * height + pos_x - 1) * width + (pos_y + 1)];
		s10 = (float)in_image[(2 * height + pos_x - 1) * width + (pos_y)];
		s20 = (float)in_image[(2 * height + pos_x - 1) * width + (pos_y - 1)];
		s01 = (float)in_image[(2 * height + pos_x) * width + (pos_y + 1)];
		s21 = (float)in_image[(2 * height + pos_x) * width + (pos_y - 1)];
		s02 = (float)in_image[(2 * height + pos_x + 1) * width + (pos_y + 1)];
		s12 = (float)in_image[(2 * height + pos_x + 1) * width + (pos_y)];
		s22 = (float)in_image[(2 * height + pos_x + 1) * width + (pos_y - 1)];
		sx = s00 + 2 * s10 + s20 - (s02 + 2 * s12 + s22);
		sy = s00 + 2 * s01 + s02 - (s20 + 2 * s21 + s22);
		float b = sqrt(sx * sx + sy * sy);
		//b = 0.0f;

		out_image[pos_x * width + pos_y] = (unsigned char)(r);
		out_image[(height + pos_x) * width + pos_y] = (unsigned char)(g);
		out_image[(height * 2 + pos_x) * width + pos_y] = (unsigned char)(b);

	}
}
void calledgedetection(dim3 blocks, dim3 threads, unsigned char* out_image, unsigned char* d_input, int height, int width)
{
	edgedetection <<< blocks, threads >>> (out_image, d_input, height, width);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
}

__global__
void sharpen(unsigned char* out_image, float factor, unsigned char* input_imag, int height, int width)
{
	int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
	int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (pos_x >= 1 && pos_x < width - 1 && pos_y >= 1 && pos_y < height - 1)
	{

		unsigned char r = input_imag[pos_y * width + pos_x];
		unsigned char g = input_imag[(height + pos_y) * width + pos_x];
		unsigned char b = input_imag[(height * 2 + pos_y) * width + pos_x];

		float sumR = float(0.0f);
		float sumG = float(0.0f);
		float sumB = float(0.0f);
		float filter[] = { 0,-1, 0, -1, 4, -1, 0, -1, 0 };

		for (int i = (-1); i <= 1; i++)
			for (int j = (-1); j <= 1; j++)
			{
				int idx = (i + 1) * (2 * 1 + 1) + j + 1;
				if (pos_x + i > 0 && pos_y + j > 0 && pos_x + i <= width && pos_y + j <= height)
				{
					sumR += (float)input_imag[(pos_y + j) * width + (pos_x + i)] * filter[idx];
					sumG += (float)input_imag[(height + (pos_y + j)) * width + (pos_x + i)] * filter[idx];
					sumB += (float)input_imag[(height * 2 + (pos_y + j)) * width + (pos_x + i)] * filter[idx];
				}
			}

		out_image[pos_y * width + pos_x] = (unsigned char)sumR * factor + r;

		out_image[(height + pos_y) * width + pos_x] = (unsigned char)sumG * factor + g;

		out_image[(height * 2 + pos_y) * width + pos_x] = (unsigned char)sumB * factor + b;

	}
}
void callsharpen(dim3 blocks, dim3 threads, unsigned char* out_image, float sharpenfactor, unsigned char* input_image, int height, int width)
{
	sharpen << < blocks, threads >> > (out_image, sharpenfactor, input_image, height, width);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
}

__global__
void constantGauss(unsigned char* out_image, unsigned char* in_image, int height, int width) {
	int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
	int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (pos_x < width && pos_y < height) {

		float R = float(0.0f);
		float G = float(0.0f);
		float B = float(0.0f);
		int K_2 = KERNEL_SIZE / 2;
		float size = KERNEL_SIZE * KERNEL_SIZE;

		for (int i = (-K_2); i <= K_2; i++)
			for (int j = (-K_2); j <= K_2; j++)
			{
				int idx = (i + K_2) * KERNEL_SIZE + j + K_2; 
				if (pos_x + i > 0 && pos_y + i > 0 && pos_x + j <= width && pos_y + i <= height)
				{
					R += (float)in_image[(pos_x + i) * width + (pos_y + j)] * Ma[idx];
					G += (float)in_image[(height + (pos_x + i)) * width + (pos_y + j)] * Ma[idx];
					B += (float)in_image[(height * 2 + (pos_x + i)) * width + (pos_y + j)] * Ma[idx];
				}
			}

		R = R > 255 ? 255 : R < 0 ? 0 : R;
		G = G > 255 ? 255 : G < 0 ? 0 : G;
		B = B > 255 ? 255 : B < 0 ? 0 : B;
		//R = R / size;
		//G = G / size;
		//B = B / size;
		out_image[pos_x * width + pos_y] = (unsigned char)(R);
		out_image[(height + pos_x) * width + pos_y] = (unsigned char)(G);
		out_image[(height * 2 + pos_x) * width + pos_y] = (unsigned char)(B);

	}

}
void callconstantGauss(dim3 blocks, dim3 threads, unsigned char* out_image, unsigned char* d_input, int height, int width)
{
	constantGauss <<< blocks, threads >>> (out_image, d_input, height, width);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
}


__global__
void sharedGauss(unsigned char* out_image, unsigned char* in_image, float* conv_kernel, int height, int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y * TILE_SIZE + ty;
	int col_o = blockIdx.x * TILE_SIZE + tx;

	int KERNEL_SIZE_2 = KERNEL_SIZE / 2;

	int row_i = row_o - KERNEL_SIZE_2;
	int col_i = col_o - KERNEL_SIZE_2;

	// part2
	__shared__ float N_ds[BLOCK_SIZE][BLOCK_SIZE][3];
	if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
	{
		N_ds[ty][tx][0] = (float)in_image[row_i * width + col_i];
		N_ds[ty][tx][1] = (float)in_image[(height + row_i) * width + col_i];
		N_ds[ty][tx][2] = (float)in_image[(height * 2 + row_i) * width + col_i];
	}
	else
	{
		N_ds[ty][tx][0] = 0.0f;
		N_ds[ty][tx][1] = 0.0f;
		N_ds[ty][tx][2] = 0.0f;
	}

	__syncthreads();

	//part3
	if (ty < TILE_SIZE && tx < TILE_SIZE)
	{
		float R = float(0.0f);
		float G = float(0.0f);
		float B = float(0.0f);
		for ( int i = 0; i < KERNEL_SIZE; i++)
			for ( int j = 0; j < KERNEL_SIZE; j++)
			{
				int convidx = i * KERNEL_SIZE + j;
				R += (float)N_ds[i + ty][j + tx][0] * conv_kernel[convidx];
				G += (float)N_ds[i + ty][j + tx][1] * conv_kernel[convidx];
				B += (float)N_ds[i + ty][j + tx][2] * conv_kernel[convidx];

			}
		R = R > 255 ? 255 : R < 0 ? 0 : R;
		G = G > 255 ? 255 : G < 0 ? 0 : G;
		B = B > 255 ? 255 : B < 0 ? 0 : B;

		if (row_o < height && col_o < width)
		{
			out_image[row_o * width + col_o] = (unsigned char)(R);
			out_image[(height + row_o) * width + col_o] = (unsigned char)(G);
			out_image[(height * 2 + row_o) * width + col_o] = (unsigned char)(B);
		}
	}
}

void callsharedGauss(dim3 blocks, dim3 threads, unsigned char* out_image, unsigned char* d_input, float* conv_kernel, int height, int width)
{
	sharedGauss <<< blocks, threads >> > (out_image, d_input, conv_kernel, height, width);
	// sharedGauss << <blocks, threads, TILE_SIZE* TILE_SIZE * 3 * sizeof(unsigned char) >> > (out_image, d_input, conv_kernel, height, width);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
}

