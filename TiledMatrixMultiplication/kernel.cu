#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/**
* The kernel function for performing tiled matrix multiplication.
* This is performed by a single CUDA thread in a CUDA block in a grid.
* 
* a = pointer to matrix a
* b = pointer to matrix b
* result = output submatrix
* 
* a is a mxm (rows x columns) matrix
* b is a mxm matrix
* result is a mxm matrix
* 
**/
__global__ void tiled_matrix_multiplication(float* a, float* b, float* result, int m) {
	
	// SHARED MEMORY IN A SINGLE CUDA BLOCK
	__shared__ float shareA[2][2];
	__shared__ float shareB[2][2];

	// 2D grid of blocks where each block represents a single matrix tile
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;

	// 2D grid of threads that compose each block
	// each thread represents a single element in the matrix
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;

	// result MATRIX LOCATION
	int row = 2 * block_y + thread_y;
	int col = 2 * block_x + thread_x;

	// Use to find matrix multiplication of 2x2 shareA and shareB
	float temp = 0;


	for (int i = 0; i < m / 2; i++) {
		shareA[thread_y][thread_x] = a[row * m + (i * 2 + thread_x)];
		shareB[thread_y][thread_x] = b[(i * 2 + thread_y) * m + col];
		
		/**
		* Ensures all threads have reached this point
		* (updated shareA and shareB) together before proceeding
		**/
		__syncthreads(); 

		for (int j = 0; j < 2; j++) {
			temp += shareA[thread_y][j] * shareB[j][thread_x];
			__syncthreads();
		}

	}

	result[row * m + col] = temp;

}

int main() {
	int m = 4; //size of matrices (m x m)

	/**
	* ======= Definitions ========
	* host = (main CPU and CPU mem)
	* device = (GPU and GPU mem)
	* ============================
	*
	* First we need to allocate memory in the host to store the
	* initial matrices. We will then copy these matrices to the
	* GPU and then call the kernel function to perform the parallel
	* processing. Once complete we will then copy the data from the
	* device (GPU) back to the host (CPU)
	*
	**/

	// Allocates and inits host matrices, pointers to
	float* host_a = (float*) malloc(m * m * sizeof(float));
	float* host_b = (float*)malloc(m * m * sizeof(float));
	float* host_result = (float*)malloc(m * m * sizeof(float));

	// Define and initialize matrices A and B
	float host_a_data[16] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
	float host_b_data[16] = { 2.0, 0.0, 1.0, 3.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 0.0 };
	memcpy(host_a, host_a_data, m * m * sizeof(float));
	memcpy(host_b, host_b_data, m * m * sizeof(float));



	// Allocating device (GPU) memory
	float* device_a;
	float* device_b;
	float* device_result;

	// Allocating mem in GPU for matrices
	cudaMalloc((void**)&device_a, m * m * sizeof(float));
	cudaMalloc((void**)&device_b, m * m * sizeof(float));
	cudaMalloc((void**)&device_result, m * m * sizeof(float));


	// cudaError_t cudaMemcpy(void* destination, const void* source, size_t count, cudaMemcpyKind kind);


	// Copy matrices from host to device
	cudaMemcpy(device_a, host_a, m * m * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, m * m * sizeof(float), cudaMemcpyHostToDevice);
	
	/**
	* dim3 is a datatype of CUDA that specifies the dimension of a
	* block (threads x threads) and a grid (blocks x blocks). Blocks
	* and grids can be 1D, 2D, or 3D. It is up to the user to specify.
	**/
	dim3 dimBlock(2, 2); // 2 x 2 threads per block
	dim3 dimGrid(m / 2, m / 2); // (m/2) x (m/2) blocks in grid


	// launch the CUDA kernel
	tiled_matrix_multiplication << <dimGrid, dimBlock >> > (device_a, device_b, device_result, m);

	// Copy result from device to host
	cudaMemcpy(host_result, device_result, m * m * sizeof(float), cudaMemcpyDeviceToHost);

	// PRINT RESULTS ==================================
	
	// Print the original host_a matrix
	printf("Matrix A:\n");
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			printf("%f ", host_a[i * m + j]);
		}
		printf("\n");
	}
	printf("\n");

	// Print the original host_b matrix
	printf("Matrix B:\n");
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			printf("%f ", host_b[i * m + j]);
		}
		printf("\n");
	}
	printf("\n");

	// Print the host_result
	printf("Result Matrix:\n");
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			printf("%f ", host_result[i * m + j]);
		}
		printf("\n");
	}

	// ================================================

	// Free device memory
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_result);

	// Free host memory
	free(host_a);
	free(host_b);
	free(host_result);

	// Freeing memory does not delete its content it just allows 
	// for it to be allocated to again within the same program

	return 0;
}