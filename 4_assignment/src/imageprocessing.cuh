#pragma once



__global__ void initArray(float* dst, float value);

__global__ void brightness(unsigned char* out_image, float brightfactor, unsigned char* in_image, int height, int width);
__global__ void contrast(unsigned char* out_image, float contrastfactor, unsigned char* in_image, int height, int width);
__global__ void saturation(unsigned char* out_image, float sharpenfactor, unsigned char* in_image, int height, int width);

void callbrightness(dim3 blocks, dim3 threads, unsigned char* out_image, float brightfactor, unsigned char* in_image, int height, int width);
void callcontrast(dim3 blocks, dim3 threads, unsigned char* out_image, float contrastfactor, unsigned char* d_input, int height, int width);
void callsaturation(dim3 blocks, dim3 threads, unsigned char* out_image, float saturationfactor, unsigned char* d_input, int height, int width);


void testCudaCall();