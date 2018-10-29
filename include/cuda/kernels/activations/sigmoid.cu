#pragma once

#include "sigmoid.cuh"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			namespace activations
			{
				__global__ void sigmoid(const float* __restrict__ A, float* __restrict__ B, size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const float val_a = A[index];
						const float val_b = 1 + expf(-1 * val_a);
						B[index] = 1 / val_b;
					}
				}
				__global__ void sigmoid(const double* __restrict__ A, double* __restrict__ B, size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const double val_a = A[index];
						const double val_b = 1 + exp(-1 * val_a);
						B[index] = 1 / val_b;
					}
				}
				__global__ void dsigmoid(const float* __restrict__ A, float* __restrict__ B, size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const float val_a = A[index];
						B[index] = (1 - val_a)  * val_a;
					}
				}
				__global__ void dsigmoid(const double* __restrict__ A, double* __restrict__ B, size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const double val_a = A[index];
						B[index] = (1 - val_a)  * val_a;
					}
				}
			}
		}
	}
}