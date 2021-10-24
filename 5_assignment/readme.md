--------------------------------------------------------------------
# CS380 GPU and GPGPU Programming
### Programming Assignment #5
### Matrix Vector Multiplication, and Reduction 
### Contacts: peter.rautek@kaust.edu.sa
--------------------------------------------------------------------

# Introduction:
In this homework you will implement vector and matrix operations.

A Conjugate Gradient Linear Systems Solver [1] will use the operations to iteratively solve a linear system Ax = b.
The skeleton of the algorithm is already provided. Your task is the implementation of the vector and matrix operations.


# Tasks: 

1. Matrix, Vector Operations and Reduction

1.a) You have to program the matrix and vector operations (matrix-vector multiplication, vector-vector operations, and reduction) on the CPU.
1.b) Implement the same operations on the GPU.

Find the comments "TASK" in the code and implement them (one CPU version and one GPU version). 
For testing a small hard-coded 4x4 matrix, and .txt files containing sparse matrices are provided. 

2. Improve the performance of the GPU implementation
You must use shared memory for the vector in the matrix-vector-multiplication!
For the reduction you must use shared memory as well (as discussed in the lecture).
Measure the performance of CPU, unoptimized and optimized GPU versions.

3. Test using Image Deblurring Method
Once the Conjugate Gradient method is working for the matrices it can be used to solve practical applications.
A simple image de-blurring application is implemented that works for a known blurring filter as described in the following:
An image is loaded and blurred with a filter kernel.
The image can be de-blurred again by solving the equation Ax = b
were x is the unknown input image, that was filtered with filter operation A such that the result is the known blurred image b.
In order to formulate the blurring operation (convolution) as a matrix multiplication, 
b and x are represented as vectors and each row of matrix A is one filter kernel for the whole image x.
This naive method is very memory inefficient (O(N^2) where N is the number of pixels). 
Your task is to test if your GPU implementations (of tasks 1. and 2.) work and to measure the performance.
Hint: Because of numerical inaccuracies for the very large matrices that occur you will have to increase the error tolerance for the CG-method to terminate. 


BONUS: 
1. Implement a sparse matrix format.
2. Modify the code to divide the image into small patches and solve for them separately. This will allow to run the de-blurring on larger input images.


References:
[1] http://en.wikipedia.org/wiki/Conjugate_gradient_method