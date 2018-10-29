#pragma once

#include "relu.cuh"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			namespace activations
			{
				__global__ void relu(const float* __restrict__ A, float* __restrict__ B, size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const float val_a = A[index];
						B[index] = fmaxf((float)0, val_a);
					}
				}
				__global__ void relu(const double* __restrict__ A, double* __restrict__ B, size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const double val_a = A[index];
						B[index] = fmax((double)0, val_a);
					}
				}
				__global__ void drelu(const float* __restrict__ A, float* __restrict__ B, size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const float val_a = A[index];
						B[index] = (val_a > 0) ? 1 : 0;
					}
				}
				__global__ void drelu(const double* __restrict__ A, double* __restrict__ B, size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const double val_a = A[index];
						B[index] = (val_a > 0) ? 1 : 0;
					}
				}
			}
		}
	}
}