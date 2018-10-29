#pragma once

#include "tanh.cuh"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			namespace activations
			{
				__global__ void tanh_(const float* __restrict__ A, float* __restrict__ B, size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const float val_a = A[index];
						B[index] = tanhf(val_a);
					}
				}
				__global__ void tanh_(const double* __restrict__ A, double* __restrict__ B, size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const double val_a = A[index];
						B[index] = tanh(val_a);
					}
				}
				__global__ void dtanh(const float* __restrict__ A, float* __restrict__ B, size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const float val_a = A[index];
						B[index] = 1 - (val_a * val_a);
					}
				}
				__global__ void dtanh(const double* __restrict__ A, double* __restrict__ B, size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const double val_a = A[index];
						B[index] = 1 - (val_a * val_a);
					}
				}
			}
		}
	}
}