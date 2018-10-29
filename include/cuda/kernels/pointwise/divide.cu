#pragma once

#include "divide.cuh"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			namespace pointwise
			{
				__global__ void divide(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, const size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const float val_a = A[index];
						const float val_b = B[index];
						C[index] = val_a / val_b;
					}
				}
				__global__ void divide(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, const size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const double val_a = A[index];
						const double val_b = B[index];
						C[index] = val_a / val_b;
					}
				}
			}
		}
	}
}