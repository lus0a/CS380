// CS 380 - GPGPU Programming
// Programming Assignment #4

// system includes
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <sstream>
#include <algorithm> 
#include <array>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "CImg.h"
#include "imageprocessing.cuh"


// query GPU functionality we need for CUDA, return false when not available
bool queryGPUCapabilitiesCUDA()
{
	// Device Count
	int devCount;

	// Get the Device Count
	cudaGetDeviceCount(&devCount);

	// Print Device Count
	printf("Device(s): %i\n", devCount);

	// TODO: query anything else you will need
	return devCount > 0;
}

void getconv(float **conv_kernel, int l){//mean filter 
	float *tmp = (float*)malloc(l*l*sizeof(float));
	for(int i=0; i<l*l; i++){
		tmp[i] = 1.0;
	}
	*(conv_kernel) = tmp;
}

void getgaussian(float **gaussian_kernel, int l){// Gaussian filter 
	float *tmp = (float*)malloc(l*l*sizeof(float));
	int half_l = l / 2;
	float sigma = 1.0f;
	float sum = 0.0f;
	for(int i=-half_l; i<half_l+1; i++)
		for(int j=-half_l; j<half_l+1; j++){
			tmp[(i+half_l)*l+j+half_l] = 1/(2*3.1415926*sigma*sigma)*exp(-1*((i*i)+(j*j))/(2*sigma*sigma));
			sum += tmp[(i + half_l) * l + j + half_l];
		}
	for (int i = 0; i < l; i++)
		for (int j = 0; j < l; j++) {
			tmp[i * l + j] /= sum;
			tmp[i * l + j] *= l * l;
		}
	*(gaussian_kernel) = tmp;
}

// entry point
int main(int argc, char** argv)
{

	// query CUDA capabilities
	if (!queryGPUCapabilitiesCUDA())
	{
		// quit in case capabilities are insufficient
		exit(EXIT_FAILURE);
	}


	testCudaCall();


	// simple example taken and modified from http://cimg.eu/
	// load image
	cimg_library::CImg<unsigned char> image("../data/images/lichtenstein_full.bmp");
	// create image for simple visualization
	cimg_library::CImg<unsigned char> visualization(512, 300, 1, 3, 0);
	const unsigned char red[] = { 255, 0, 0 };
	const unsigned char green[] = { 0, 255, 0 };
	const unsigned char blue[] = { 0, 0, 255 };

	int imgheight = image.height();
	int imgwidth = image.width();
	std::cout << imgheight << std::endl << imgwidth << std::endl;
	
	int imgproduct = imgheight*imgwidth;
	cimg_library::CImg<unsigned char> originimage = image;

	float brightnessfactor = 1.0f;
	float contrastfactor = 1.0f;
	float saturationfactor = 1.0f;
	bool smooth = false;
	int smoothsize = 3;
	float * smoothconv, * dsmoothconv;
	bool gauss = false;
	bool edgedetection = false;
	bool sharpen = false;
	unsigned char  *d_input, *d_output;
	cudaMalloc((void **)&d_input, imgproduct*3*sizeof(unsigned char));
	cudaMalloc((void **)&d_output, imgproduct*3*sizeof(unsigned char));
	cudaMemcpy(d_input, originimage, imgproduct*3*sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 threads(8,8);
	dim3 blocks(imgwidth/threads.x, imgheight/threads.y);

	// create displays 
	cimg_library::CImgDisplay inputImageDisplay(image, "click to select row of pixels");
	inputImageDisplay.move(40, 40);
	cimg_library::CImgDisplay visualizationDisplay(visualization, "intensity profile of selected row");
	visualizationDisplay.move(600, 40);
	while (!inputImageDisplay.is_closed() && !visualizationDisplay.is_closed()) {
		inputImageDisplay.wait();

		if (inputImageDisplay.button() && inputImageDisplay.mouse_y() >= 0) {
			// on click redraw visualization
			const int y = inputImageDisplay.mouse_y();
			visualization.fill(0).draw_graph(image.get_crop(0, y, 0, 0, image.width() - 1, y, 0, 0), red, 1, 1, 0, 255, 0);
			visualization.draw_graph(image.get_crop(0, y, 0, 1, image.width() - 1, y, 0, 1), green, 1, 1, 0, 255, 0);
			visualization.draw_graph(image.get_crop(0, y, 0, 2, image.width() - 1, y, 0, 2), blue, 1, 1, 0, 255, 0).display(visualizationDisplay);
		}
		else if(inputImageDisplay.is_keyB()){ // brightness up
			brightnessfactor -= 0.1;			
			brightnessfactor = brightnessfactor < 0.0 ? 0.0 : brightnessfactor;
			// float brightnessfactor =0.5;
			printf("brightness %f \n", brightnessfactor);
			cudaMemcpy(d_input, originimage, imgproduct*3*sizeof(unsigned char), cudaMemcpyHostToDevice);
			callbrightness(blocks, threads, d_output, brightnessfactor, d_input, imgheight, imgwidth);
			cudaMemcpy(image, d_output, imgproduct*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		}
		else if(inputImageDisplay.is_keyG()){ // brightness down
			brightnessfactor += 0.1;
			brightnessfactor = brightnessfactor > 1.0 ? 1.0 : brightnessfactor;
			printf("brightness %f \n", brightnessfactor);
			cudaMemcpy(d_input, originimage, imgproduct*3*sizeof(unsigned char), cudaMemcpyHostToDevice);
			callbrightness(blocks, threads, d_output, brightnessfactor, d_input, imgheight, imgwidth);
			cudaMemcpy(image, d_output, imgproduct*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		}
		else if(inputImageDisplay.is_keyH()){ // contrast up
			contrastfactor += 0.1;			
			contrastfactor = contrastfactor > 1.0 ? 1.0 : contrastfactor;
			printf("contrastfactor %f \n", contrastfactor);
			cudaMemcpy(d_input, originimage, imgproduct*3*sizeof(unsigned char), cudaMemcpyHostToDevice);
			callcontrast(blocks, threads, d_output, contrastfactor, d_input, imgheight, imgwidth);
			cudaMemcpy(image, d_output, imgproduct*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		}
		else if(inputImageDisplay.is_keyN()){ // contrast down
			contrastfactor -= 0.1;			
			contrastfactor = contrastfactor < 0.0 ? 0.0 : contrastfactor;
			printf("contrastfactor %f \n", contrastfactor);
			cudaMemcpy(d_input, originimage, imgproduct*3*sizeof(unsigned char), cudaMemcpyHostToDevice);
			callcontrast(blocks, threads, d_output, contrastfactor, d_input, imgheight, imgwidth);
			cudaMemcpy(image, d_output, imgproduct*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		}
		else if(inputImageDisplay.is_keyJ()){ // saturation up
			saturationfactor += 0.1;			
			saturationfactor = saturationfactor > 1.0 ? 1.0 : saturationfactor;
			printf("saturationfactor %f \n", saturationfactor);
			cudaMemcpy(d_input, originimage, imgproduct*3*sizeof(unsigned char), cudaMemcpyHostToDevice);
			callsaturation(blocks, threads, d_output, saturationfactor, d_input, imgheight, imgwidth);
			cudaMemcpy(image, d_output, imgproduct*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		}
		else if(inputImageDisplay.is_keyM()){ // saturation down
			saturationfactor -= 0.1;			
			saturationfactor = saturationfactor < 0.0 ? 0.0 : saturationfactor;
			printf("saturationfactor %f \n", saturationfactor);
			cudaMemcpy(d_input, originimage, imgproduct*3*sizeof(unsigned char), cudaMemcpyHostToDevice);
			callsaturation(blocks, threads, d_output, saturationfactor, d_input, imgheight, imgwidth);
			cudaMemcpy(image, d_output, imgproduct*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		}
		else if(inputImageDisplay.is_key1()){
			smoothsize += 2;
			printf("set smmoth size to  %f \n", smoothsize);
		} 
		else if (inputImageDisplay.is_keyT()) { // smooth on/off
			cudaMemcpy(d_input, originimage, imgproduct*3*sizeof(unsigned char), cudaMemcpyHostToDevice);
			if(smooth)
			{
				printf("trun off smooth\n");
				smooth = false;
			  	cudaMemcpy(d_output, d_input, imgproduct*3*sizeof(unsigned char), cudaMemcpyDeviceToDevice);
			}
			else
			{
				printf("trun on smooth\n");
				smooth = true;
				getconv(&smoothconv, smoothsize);
				cudaMalloc((void **)&dsmoothconv, smoothsize*smoothsize*sizeof(float));
				cudaMemcpy(dsmoothconv, smoothconv, smoothsize*smoothsize*sizeof(float), cudaMemcpyHostToDevice);
				callsmooth(blocks, threads, d_output, d_input, dsmoothconv, smoothsize, imgheight, imgwidth);
			}
			cudaMemcpy(image, d_output, imgproduct*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		}
		else if (inputImageDisplay.is_keyY()) { // Gaussian on/off
			cudaMemcpy(d_input, originimage, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
			if (smooth)
			{
				printf("trun off Gaussian\n");
				smooth = false;
				cudaMemcpy(d_output, d_input, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
			}
			else
			{
				printf("trun on Gaussian\n");
				smooth = true;
				getgaussian(&smoothconv, smoothsize);
				cudaMalloc((void**)&dsmoothconv, smoothsize * smoothsize * sizeof(float));
				cudaMemcpy(dsmoothconv, smoothconv, smoothsize * smoothsize * sizeof(float), cudaMemcpyHostToDevice);
				callsmooth(blocks, threads, d_output, d_input, dsmoothconv, smoothsize, imgheight, imgwidth);
			}
			cudaMemcpy(image, d_output, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		}
		else if (inputImageDisplay.is_keyE()) { // EdgeDetection
			float kernel_size = 3;
			cudaMemcpy(device_input, image1, I_size * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
			edgeDetectionCall(numBlocks, threadPerBlocks, device_output, kernel_size, device_input, I_height, I_width);
			cudaMemcpy(image, device_output, I_size * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		}
		else if(inputImageDisplay.is_keyESC()){ // quit
			break;
		}
		inputImageDisplay.display(image);
	}

	// save test output image
	visualization.save("./test_output.bmp");

	return EXIT_SUCCESS;
}

