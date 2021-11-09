#pragma once
#define KERNEL_SIZE 9
#define TILE_SIZE 32
#define BLOCK_SIZE (TILE_SIZE + KERNEL_SIZE - 1)

__constant__ float Ma[KERNEL_SIZE* KERNEL_SIZE];

__global__ void initArray(float* dst, float value);

__global__ void brightness(unsigned char* out_image, float brightfactor, unsigned char* in_image, int height, int width);
__global__ void contrast(unsigned char* out_image, float contrastfactor, unsigned char* in_image, int height, int width);
__global__ void saturation(unsigned char* out_image, float sharpenfactor, unsigned char* in_image, int height, int width);
__global__ void smooth(unsigned char* out_image, unsigned char* in_image,  float *conv_kernel, int length, int height, int width);
__global__ void Unrollsmooth(unsigned char* out_image, unsigned char* in_image,  float *conv_kernel, int length, int height, int width);
__global__ void edgedetection(unsigned char* out_image, unsigned char* in_image, int height, int width);
__global__ void sharpen(unsigned char* out_image, float sharpenfactor, unsigned char* in_image, int height, int width);
__global__ void constantGauss(unsigned char* out_image, unsigned char* in_image, int height, int width);
__global__ void sharedGauss(unsigned char* out_image, unsigned char* in_image, float *conv_kernel, int height, int width);
//__global__ void profilingconvlayer(unsigned char* out_image, unsigned char* in_image, float *conv_kernel, int length, int height, int width);
//__global__ void convlayer(float* out_image, unsigned char* in_image, float *conv_kernel, int length, int height, int width);
//__global__ void getminmaxonRGBalongy(float* inputformax, float*intputformin, float* minmaxoutput, int height, int width);
//__global__ void getminmaxonRGBalongx(float* inputformax, float*intputformin, int height, int width);
//__global__ void NormalizationOutput(unsigned char* out_image, float*input, float* minmaxoutput, int height, int width);
//__global__ void floattoimage(unsigned char* out_image, float*input, int height, int width);
//__global__ void profilingconvlayer2(unsigned char* out_image, unsigned char* in_image, float *conv_kernel, int length, int channel, int height, int width);

void callbrightness(dim3 blocks, dim3 threads, unsigned char* out_image, float brightfactor, unsigned char* in_image, int height, int width);
void callcontrast(dim3 blocks, dim3 threads, unsigned char* out_image, float contrastfactor, unsigned char* d_input, int height, int width);
void callsaturation(dim3 blocks, dim3 threads, unsigned char* out_image, float saturationfactor, unsigned char* d_input, int height, int width);
void callsmooth(dim3 blocks, dim3 threads, unsigned char* out_image,  unsigned char* d_input, float *conv_kernel, int length,int height, int width);
void callUnrollsmooth(dim3 blocks, dim3 threads, unsigned char* out_image,  unsigned char* d_input, float *conv_kernel, int length,int height, int width);
void calledgedetection(dim3 blocks, dim3 threads, unsigned char* out_image, unsigned char* d_input, int height, int width);
void callsharpen(dim3 blocks, dim3 threads, unsigned char* out_image, float sharpenfactor, unsigned char* d_input, int height, int width);
void callconstantGauss(dim3 blocks, dim3 threads, unsigned char* out_image, unsigned char* d_input, int height, int width);
void callsharedGauss(dim3 blocks, dim3 threads, unsigned char* out_image, unsigned char* d_input, float *conv_kernel, int height, int width);


void testCudaCall();