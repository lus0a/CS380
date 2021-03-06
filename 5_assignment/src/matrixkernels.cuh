#pragma once


enum VEC_OP{

	NONE = -1,
	CL_ADD = 0,
	CL_SUB = 1,
	CL_MULT = 2,
	CL_MAX = 3,
	CL_MIN = 4,
	CL_ABS = 5
};


// vector operation: x = fac0*a op fac1*b
__global__ void _gpu_vector_op_( int op, float fac0, float fac1, float *a, float *b, float *x, int dim );

// matrix vector multiplication: x = A*b op c
__global__ void _gpu_matrix_vector_( int op, float *A, float *b, float *c, float *x, int dim );

__global__ void _gpu_vector_reduce_(int op, float* g_data, int n);
//__global__ void _gpu_vector_reduce_(float* g_data, int n);
	
// d_x = SUM[d_a * d_b]
float gpuReduceSUM( float* d_a, float *d_b, float* d_x, int dim, int nBlocks, int nThreads );

// x = A*a
extern "C"
void multiplyMatrixVector( float *h_A, float *h_a, float *h_x, int dim );

extern "C"
void computeConjugateGradientGPU( float *h_A, float *h_b, float *h_x, int dim, float errorTolerance);
