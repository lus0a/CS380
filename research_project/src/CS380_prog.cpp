// CS 380 - GPGPU Programming
// Programming Assignment #5


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
#include <assert.h> 

#include "timer.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "CImg.h"
#include "matrixkernels.cuh"


#include <fstream>

float W=1.0, H=1.0;

// get A matrix from 5 point stencil
void get_A(float *&A, int dim_grid, int dim_block)
{ 	//dim_grid = n-2
	//dim_block = m-2
	A = new float[dim_grid * dim_grid * dim_block * dim_block];
	memset(A, 0, dim_grid*dim_grid*dim_block*dim_block * sizeof(float));
	for (int ni=0; ni<dim_grid; ni++)
	{
		if (ni>0 && ni<dim_grid-1)
		{
			for (int mi=0; mi<dim_block; mi++)
			{   
				//D
				if (mi>0 && mi<dim_block-1)
				{
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi] = 4;
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi+1] = -1;
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi-1] = -1;
				}
				else if (mi==0)
				{
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi] = 4;
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi+1] = -1;
				}
				else if (mi==dim_block-1)
				{
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi] = 4;
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi-1] = -1;
				}
				//Left -I
				A[ni*dim_grid*dim_block*dim_block + (ni-1)*dim_block + mi*dim_grid*dim_block+mi] = -1;
				//Right -I
				A[ni*dim_grid*dim_block*dim_block + (ni+1)*dim_block + mi*dim_grid*dim_block+mi] = -1;
			}
		}
		else if (ni==0)
		{
			for (int mi=0; mi<dim_block; mi++)
			{   
				//D
				if (mi>0 && mi<dim_block-1)
				{
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi] = 4;
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi+1] = -1;
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi-1] = -1;
				}
				else if (mi==0)
				{
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi] = 4;
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi+1] = -1;
				}
				else if (mi==dim_block-1)
				{
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi] = 4;
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi-1] = -1;
				}
				//Right -I
				A[ni*dim_grid*dim_block*dim_block + (ni+1)*dim_block + mi*dim_grid*dim_block+mi] = -1;
			}	
		}
		else if (ni==dim_grid-1)
		{
			for (int mi=0; mi<dim_block; mi++)
			{   
				//D
				if (mi>0 && mi<dim_block-1)
				{
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi] = 4;
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi+1] = -1;
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi-1] = -1;
				}
				else if (mi==0)
				{
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi] = 4;
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi+1] = -1;
				}
				else if (mi==dim_block-1)
				{
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi] = 4;
					A[ni*dim_grid*dim_block*dim_block + ni*dim_block + mi*dim_grid*dim_block+mi-1] = -1;
				}
				//Left -I
				A[ni*dim_grid*dim_block*dim_block + (ni-1)*dim_block + mi*dim_grid*dim_block+mi] = -1;
			}	
		}
	}
}

float fxy(float x, float y)
{
	return exp(x + y/2)*1.25;
	//return 0;
}

float exactu(float x, float y)
{
	return exp(x + y/2);
}

void get_b(float*& b, int dim_grid, int dim_block)
{
	//dim_grid = n-2
	//dim_block = m-2
	b = new float[dim_grid * dim_block];
	float h=H/(dim_grid+1);
	memset(b, 0, dim_grid * dim_block * sizeof(float));
	for (int ni = 0; ni < dim_grid; ni++)
	{
		//inner line
		if (ni > 0 && ni < dim_grid - 1)
		{
			for (int mi = 0; mi < dim_block; mi++)
			{
				b[ni * dim_grid + mi] = -h * h * fxy(float(mi+1)/ float(dim_block+1), float(ni+1)/ float(dim_grid+1));
				//left element
				if (mi == 0)
				{
					b[ni * dim_grid + mi] += exactu(float(mi)/ float(dim_block+1), float(ni+1)/ float(dim_grid+1));
				}
				//right element
				else if (mi == dim_block - 1)
				{
					b[ni * dim_grid + mi] += exactu(float(mi+2)/ float(dim_block+1), float(ni+1)/ float(dim_grid+1));
				}
			}
		}
		//bot line
		if (ni == 0)
		{
			for (int mi = 0; mi < dim_block; mi++)
			{
				b[ni * dim_grid + mi] = -h * h * fxy(float(mi+1)/ float(dim_block+1), float(ni+1)/ float(dim_grid+1));
				//inner elements
				if (mi > 0 && mi < dim_block - 1)
				{
					b[ni * dim_grid + mi] += exactu(float(mi + 1) / float(dim_block + 1), float(ni)/ float(dim_grid + 1));
				}
				//left element
				else if (mi == 0)
				{
					b[ni * dim_grid + mi] += exactu(float(mi)/ float(dim_block+1), float(ni+1)/ float(dim_grid+1)) + exactu(float(mi+1)/ float(dim_block+1), float(ni)/ float(dim_grid+1));
				}
				//right element
				else if (mi == dim_block - 1)
				{
					b[ni * dim_grid + mi] += exactu(float(mi+2)/ float(dim_block+1), float(ni+1)/ float(dim_grid+1)) + exactu(float(mi+1)/ float(dim_block+1), float(ni)/ float(dim_grid+1));
				}
			}
		}
		//top line
		if (ni == dim_grid - 1)
		{
			for (int mi = 0; mi < dim_block; mi++)
			{
				b[ni * dim_grid + mi] = -h * h * fxy(float(mi+1)/ float(dim_block+1), float(ni+1)/ float(dim_grid+1));
				//inner elements
				if (mi > 0 && mi < dim_block - 1)
				{
					b[ni * dim_grid + mi] += exactu(float(mi + 1) / float(dim_block + 1), float(ni + 2) / float(dim_grid + 1));
				}
				//left element
				else if (mi == 0)
				{
					b[ni * dim_grid + mi] += exactu(float(mi)/ float((dim_block+1)), float((ni+1))/ float((dim_grid+1))) + exactu(float((mi+1))/ float(dim_block+1), float(ni+2)/ float(dim_grid+1));
				}
				//right element
				else if (mi == dim_block - 1)
				{
					b[ni * dim_grid + mi] += exactu(float(mi+2)/ float(dim_block+1), float(ni+1)/ float(dim_grid+1)) + exactu(float(mi+1)/ float(dim_block+1), float(ni+2)/ float(dim_grid+1));
				}
			}
		}
	}
}


enum MAJORITY {
	ROW_MAJOR = 0,
	COLUMN_MAJOR = 1
};


// query GPU functionality we need for CUDA, return false when not available
bool queryGPUCapabilitiesCUDA()
{
	// Device Count
	int devCount;

	// Get the Device Count
	cudaGetDeviceCount(&devCount);

	// Print Device Count
	printf("Device(s): %i\n", devCount);

	// query anything else you will need
	return devCount > 0;
}



// compute and print 
// sum, average and maximum 
// of error: (abs(A*x-b))
void computeResultError(float *h_A, float *h_b, float *h_x, unsigned int dim)
{
	float* h_e0 = new float[dim];
	multiplyMatrixVector(h_A, h_x, h_e0, dim);

	float sum_abs_diff_0 = 0;
	float max_abs_diff_0 = 0;

	for (unsigned int i = 0; i < dim; i++) {

		float abs_diff_0 = h_e0[i] - h_b[i];
		abs_diff_0 = abs_diff_0 > 0.0f ? abs_diff_0 : -abs_diff_0;

		if (abs_diff_0 > max_abs_diff_0)
		{
			max_abs_diff_0 = abs_diff_0;
		}
		sum_abs_diff_0 += abs_diff_0;
	}

	
	std::cout << "errors:\n";
	

	std::cout << "sum  | " << sum_abs_diff_0 << "\navgs | " << sum_abs_diff_0 / dim << "\nmax  | " << max_abs_diff_0 << std::endl << std::endl;

	delete[] h_e0;
}


// loading matrix files 
bool readMatrix(char* filename, float* &matrix, unsigned int *dim = NULL, int majority = ROW_MAJOR)
{
	unsigned int w, h, x, y, num_entries;

	float val;

	std::ifstream file(filename);

	if (file)
	{
		file >> h >> w >> num_entries;
		std::cout << w << " " << h << " " << num_entries << "\n";

		assert(w == h || w == 1 || h == 1);

		if (dim != NULL) {
			*dim = w;
			if (h > w) { *dim = h; }
		}

		matrix = new float[w * h];
		memset(matrix, 0, num_entries * sizeof(float));

		unsigned int i;
		for (i = 0; i < num_entries; i++) {

			if (file.eof()) break;

			file >> y >> x >> val;

			if (majority == ROW_MAJOR) {

				matrix[w * y + x] = val;

			}
			else if (majority == COLUMN_MAJOR) {

				matrix[h * x + y] = val;
			}
		}
		file.close();

		if (i == num_entries)
			std::cout << "\nFile read successfully\n";
		else
			std::cout << "\nFile read successfully but seems defective:\n num entries read = " << i << ", entries epected = " << num_entries << "\n";
	}
	else {
		std::cout << "Unable to open file\n";
		return false;
	}

	return true;
}


// CPU implementation


// vector operation: x = fac0*a op fac1*b
void vector_op(int op, float fac0, float fac1, float* a, float* b, float* x, int dim)
{
	/* TASK 1.1a: implement the elementwise vector operations

		x = fac0 * a (op) fac1 * b

		with op = {+,-,*, NONE}.
		NONE means x = fac0 * a   (b might be NULL)
	*/
	switch(op){
		case(-1): // NONE
			for (int i=0; i<dim; i++)
			{
				x[i] = a[i] * fac0;
			}
			break;
		case(0):  // ADD 
			for (int i=0; i<dim; i++)
			{
				x[i] = a[i] * fac0 + b[i] * fac1;
			}
			break;
		case(1):  // SUB 
			for (int i=0; i<dim; i++)
			{
				x[i] = a[i] * fac0 - b[i] * fac1;
			}
			break;
		case(2):  // MULT
			for (int i=0; i<dim; i++)
			{
				x[i] = a[i] * fac0 * b[i] * fac1;
			}
			break;
	}
}




// matrix vector multiplication: x = A*b op c
void matrix_vector(int op, float* A, float* b, float* c, float* x, int dim)
{
	/* TASK 1.2a: implement the matrix vector multiplication on the CPU

		x = A * b (op) c
	
		with op = {+,-,*,NONE}.
		NONE means x = A * b     (c might be NULL)
	*/
	for(int j=0; j<dim; j++){
		float out = 0.0;

		for(int i=0; i<dim; i++){
			out += A[j * dim + i] * b[i];
		}
		switch(op){
			case(-1): // NONE
				x[j] = out;
				break;
			case(0):  // ADD 
				x[j] = out + c[j];
				break;
			case(1):  // SUB 
				x[j] = out - c[j];
				break;
			case(2):  // MULT
				x[j] = out * c[j];
				break;
		}
	}
	
}




// returns SUM[d_a * d_b]
float reduceSUM(float* d_a, float* d_b, int dim) {

	/* TASK 1.3a: implement the vector multiplication and sum reduction
		returns SUM[d_a * d_b] 
	*/

	float sum = 0.0f;
	for (int i=0; i<dim; i++){
		sum += d_a[i]*d_b[i];
	}
	return sum;
}


void computeConjugateGradientCPU(float* h_A, float* h_b, float* h_x, int dim, float errorTolerance)
{
	// computeConjugateGradientCPU(h_A, h_b, h_x_reference, dim, errorTolerance);
	float alpha, beta, rho = 0;
	float* h_r;
	float* h_p;
	float* h_q;

	h_r = new float[dim];
	h_p = new float[dim];
	h_q = new float[dim];
	
	// init CG
	// ALGORITHM: r_0 = b-Ax_0
	// r_0 = Ax_0 - b

	matrix_vector(CL_SUB, h_A, h_x, h_b, h_r, dim);

	//float rhob_cpu = reduceSUM(h_b, h_b, dim);
	//printf("\n r_cpu is %f\n", rhob_cpu);

	// r_0 = -r_0
	vector_op(NONE, -1.0f, 0.0f, h_r, NULL, h_r, dim);

	// p_0 = r_0
	vector_op(NONE, 1.0f, 0.0f, h_r, NULL, h_p, dim);
	
	// CG needs max dim iterations
	int i = 0;
	float minRho = 1000000000;
	
	for (i = 0; i < dim; i++) {

		// rho_k = sum(r_k * r_k)
		rho = reduceSUM(h_r, h_r, dim);
		
		if (minRho > rho) {
			minRho = rho;
		}
		
		//std::cout << "iteration #" << i << ", with rho_cpu = " << rho << "          " << '\r' << std::endl;

		// check here for criterion
		if (rho < errorTolerance) {
			break;
		}

		// q_k = A*p_k
		matrix_vector(NONE, h_A, h_p, NULL, h_q, dim);

		// alpha_k = rho_k / sum(p_k * q_k)
		alpha = rho / reduceSUM(h_p, h_q, dim);

		// x_(k+1) = x_k + alpha_k * p_k
		vector_op(CL_ADD, 1.0f, alpha, h_x, h_p, h_x, dim);

		// r_(k+1) = r_k + (-alpha_k * q_k)
		vector_op(CL_ADD, 1.0f, -alpha, h_r, h_q, h_r, dim);

		// beta_k = sum(r_(k+1) * r_(k+1)) / rho_k
		beta = reduceSUM(h_r, h_r, dim) / rho;

		// p_(k+1) = r_(k+1) + beta_k * p_k
		vector_op (CL_ADD, 1.0f, beta, h_r, h_p, h_p, dim);
	}

	rho = reduceSUM(h_r, h_r, dim);

	printf("\nSolution found at iteration #%d, with rho = %f\n", i, rho);
	printf("\nminrho was %f\n", minRho);

	delete[] h_r;
	delete[] h_p;
	delete[] h_q;
}

void computeGradientDescentCPU(float* h_A, float* h_b, float* h_x, int dim, float errorTolerance)
{
	// computeGradientDescentCPU(h_A, h_b, h_x_reference, dim, errorTolerance);
	float alpha, beta, rho = 0;
	float* h_r;
	float* h_q;

	h_r = new float[dim];
	h_q = new float[dim];
	
	// init CG
	// ALGORITHM: r_0 = b-Ax_0
	// r_0 = Ax_0 - b

	matrix_vector(CL_SUB, h_A, h_x, h_b, h_r, dim);

	//float rhob_cpu = reduceSUM(h_b, h_b, dim);
	//printf("\n r_cpu is %f\n", rhob_cpu);

	// r_0 = -r_0
	vector_op(NONE, -1.0f, 0.0f, h_r, NULL, h_r, dim);
	
	// CG needs max dim iterations
	int i = 0;
	float minRho = 1000000000;
	
	for (i = 0; i < dim; i++) {

		// rho_k = sum(r_k * r_k)
		rho = reduceSUM(h_r, h_r, dim);
		
		if (minRho > rho) {
			minRho = rho;
		}
		
		// q_k = A*r_k
		matrix_vector(NONE, h_A, h_r, NULL, h_q, dim);

		// alpha_k = rho_k / sum(r_k * q_k)
		alpha = rho / reduceSUM(h_r, h_q, dim);

		// x_(k+1) = x_k + alpha_k * r_k
		vector_op(CL_ADD, 1.0f, alpha, h_x, h_r, h_x, dim);

		//std::cout << "iteration #" << i << ", with rho_cpu = " << rho << "          " << '\r' << std::endl;
		
		// check here for criterion
		if (rho < errorTolerance) {
			break;
		}

		// r_(k+1) = r_k + (-alpha_k * q_k)
		vector_op(CL_ADD, 1.0f, -alpha, h_r, h_q, h_r, dim);
	}

	rho = reduceSUM(h_r, h_r, dim);

	printf("\nSolution found at iteration #%d, with rho = %f\n", i, rho);
	printf("\nminrho was %f\n", minRho);

	delete[] h_r;
	delete[] h_q;
}

// entry point
int main(int argc, char** argv)
{
	unsigned int dim;
	float* h_A = NULL;
	float* h_b = NULL;
	float* h_exactb = NULL;
	float* h_CG_x = NULL;
	float* h_CG_x_reference = NULL;
	float* h_GD_x = NULL;
	float* h_GD_x_reference = NULL;

	/**************************************************************
	get the conjugate gradient method first running with the
	test matrices. afterwards change the parameter matrixSet to 0
	and try with different input images!
	***************************************************************/
	//float h = 0.2;
	//float h = 0.1;
	//float h = 0.05;
	float h = 0.025;

	int dimn = 1 / h - 1;
	int dimm = dimn;
	get_A(h_A, dimn, dimm);
	//for (int i = 0; i < dimn*dimm; i++)
	//{
	//	for (int j = 0; j < dimn * dimm; j++)
	//	{
	//		std::cout << h_A[i * dimn * dimm + j] << ' ';
	//	}
	//	std::cout<<std::endl;
	//}
	
	get_b(h_b, dimn, dimm);
	//for (int i = 0; i < dimn*dimm; i++)
	//{
	//	std::cout << h_b[i] << ' ';
	//}
	
	// PARAMETERS: 
	int matrixSet = dimn * dimm;
	dim = matrixSet;
	//matrixSet = 0;			// set this to 0 for the image deblurring; 
	//matrixSet = 16;		// set this to 16, 64, 200 for the other matrices
	//matrixSet = 64;										
	//matrixSet = 200;							

	//bool success = true;

	// query CUDA capabilities
	if (!queryGPUCapabilitiesCUDA())
	{
		// quit in case capabilities are insufficient
		exit(EXIT_FAILURE);
	}

	// init CUDA
	cudaSetDevice(0);

	if (matrixSet != 0) {
		// matrix application
		// init the solution to a vector of 0's
		h_CG_x = new float[dim];
		h_CG_x_reference = new float[dim];
		memset(h_CG_x, 0, dim * sizeof(float));
		memset(h_CG_x_reference, 0, dim * sizeof(float));

		h_GD_x = new float[dim];
		h_GD_x_reference = new float[dim];
		memset(h_GD_x, 0, dim * sizeof(float));
		memset(h_GD_x_reference, 0, dim * sizeof(float));

		h_exactb = new float[dim];
		memset(h_exactb, 0, dim * sizeof(float));
		// find h_x where h_A * h_x = h_b
		float errorTolerance = 0.0000001f * dim;
		
		//float rhob_cpu = reduceSUM(h_x_reference, h_x_reference, dim);
		//printf("\n rhob_cpu is %f\n", rhob_cpu);
		//float rhob_gpu = reduceSUM(h_x, h_x, dim);
		//printf("\n rhob_gpu is %f\n", rhob_gpu);

		StartTimer();
		computeConjugateGradientCPU(h_A, h_b, h_CG_x_reference, dim, errorTolerance);
		double t = GetTimer();
		std::cout << "CPU elapsed time of Conjugate Gradient method: " << t << "ms" << std::endl;
		StartTimer();
		computeConjugateGradientGPU(h_A, h_b, h_CG_x, dim, errorTolerance);
		t = GetTimer();
		std::cout << "GPU elapsed time of Conjugate Gradient method: " << t << "ms" << std::endl;
		// compute the error
		std::cout << "CPU" << std::endl;
		computeResultError(h_A, h_b, h_CG_x_reference, dim);
		std::cout << "GPU" << std::endl;
		computeResultError(h_A, h_b, h_CG_x, dim);

		StartTimer();
		computeGradientDescentCPU(h_A, h_b, h_GD_x_reference, dim, errorTolerance);
		t = GetTimer();
		std::cout << "CPU elapsed time of Gradient Descent method: " << t << "ms" << std::endl;
		StartTimer();
		computeGradientDescentGPU(h_A, h_b, h_GD_x, dim, errorTolerance);
		t = GetTimer();
		std::cout << "GPU elapsed time of Gradient Descent method: " << t << "ms" << std::endl;
		// compute the error
		std::cout << "CPU" << std::endl;
		computeResultError(h_A, h_b, h_GD_x_reference, dim);
		std::cout << "GPU" << std::endl;
		computeResultError(h_A, h_b, h_GD_x, dim);

		int dim_grid = dimn;
		int dim_block = dimm;
		for (int ni = 0; ni < dim_grid; ni++)
		{
			for (int mi = 0; mi < dim_block; mi++)
			{
				h_exactb[ni * dim_grid + mi] = exactu(float(mi + 1)/ float(dim_block + 1), float(ni + 1) / float(dim_grid + 1));
			}
		}
		std::cout << "method_error" << std::endl;
		computeResultError(h_A, h_b, h_exactb, dim);

	}
	
	delete[] h_A;
	delete[] h_CG_x;
	delete[] h_GD_x;

	if (h_CG_x_reference != NULL) {
		delete[] h_CG_x_reference;
	}
	if (h_GD_x_reference != NULL) {
		delete[] h_GD_x_reference;
	}

	if (h_b != NULL) {
		delete[] h_b;
	}
	
	return EXIT_SUCCESS;
}

