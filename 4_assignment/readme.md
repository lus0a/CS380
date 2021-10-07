--------------------------------------------------------------------
# CS380 GPU and GPGPU Programming
### Programming Assignment #4
### Image Processing with CUDA
### Contacts: peter.rautek@kaust.edu.sa
--------------------------------------------------------------------

# Introduction:
In this assignment you will implement several image processing operations using CUDA.

1. Image Processing with CUDA
- implement the pixel-wise image processing operations: brightness, contrast, and saturation (Chapter 19.5. [1]).
- implement the convolution based operations: smoothing with mean filter, smoothing with Gaussian filter, edge detection, and sharpening (Chapter 19.7. [1])
- implement different convolution kernels (5x5, 7x7, 9x9, ... n x n) for the smoothing operation. Find out what is the largest possible kernel size your code can successfully run. 

Use a c-style array as input to the CUDA kernels. 
cudaMalloc is used to allocate memory and cudaMemcpy is used to initialize device memory. 
You can (i) either re-use your framework that you have developed so far, (ii) start completely from scratch, or (iii) use the provided framework.

2. Export Images
Either use a library to export the images to a common file format on disk or visualize them on screen using the provided CImg library [3] or OpenGL.

3. Profiling
Measure the time it takes to apply the image processing operations (See: 'Timing using CUDA Events' at [2]). 
Measure the time of the CUDA kernel only (without memory initialization, transfer, ...)!
3.a) Run the Gaussian smoothing filter with increasing kernel size (3x3, .. n x n) on a image of fixed size (e.g., 1024x1024) to analyze the scaling behavior 
3.b) Run a fixed size smoothing filter (e.g., 25x25) on randomly initialized images of increasing size (32x32, ..., 128x128, ..., 1024x1024, ...) to analyze the scaling behavior.

4. Implement the convolution for small filter kernels (e.g., 9x9) using shared memory on the GPU. 
Think about how to most effectively share computations between threads. 
In Chapter 7.6 'Tiled 2D Convolution With Halo Cells' of 'Massively Parallel Processors, 3rd Edition' [4], a simple method to benefit from faster access times of shared memory is described.

5. Benchmark different methods (global vs. local size, constant memory, shared memory, loop unrolling, ...) 
5.a) Benchmark for different global/local sizes
5.b) Benchmark with and without the use of constant memory (see Chapter 7 [4])
5.c) Benchmark with and without the use of shared memory
5.d) Document the attempts to make it faster (why did it (not) become faster?).

6. Submit your program and a report including result images and profiling results for the different image processing operations.

# BONUS: 

Implement a single-threaded and/or multi-threaded CPU version of the convolution. Benchmark it and compare performance with the GPU implementation.

# References and Acknowledgments:
[1] OpenGL Shading Language, by Rost, Randi J; Licea-Kane, Bill, 2010:
https://learning.oreilly.com/library/view/opengl-shading-language/9780321669247/ch19.html

[2] NVIDIA - CUDA Performance Profiling: https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/

[3] CImg library: http://cimg.eu/

[4] Chapter 7.6 'Tiled 2D Convolution With Halo Cells' in 'Massively Parallel Processors, 3rd Edition':
https://learning.oreilly.com/library/view/Programming+Massively+Parallel+Processors,+3rd+Edition/9780128119877/xhtml/chp007.xhtml#s0020

The provided textures are from http://www.grsites.com/
