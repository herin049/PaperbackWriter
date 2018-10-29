#pragma once

#include "reciprocal.cuh"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			__global__ void reciprocal(float* __restrict__ A, const float value, const size_t size)
			{
				const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
				const size_t stride = blockDim.x * gridDim.x;
				for (size_t index = grid; index < size; index += stride)
				{
					const float val_a = A[index];
					if(value != 0)
					{ 
						A[index] = val_a / value;
					}
				}
			}
			__global__ void reciprocal(double* __restrict__ A, const double value, const size_t size)
			{
				const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
				const size_t stride = blockDim.x * gridDim.x;
				for (size_t index = grid; index < size; index += stride)
				{
					const double val_a = A[index];
					A[index] = val_a / value;
				}
			}
		}
	}
}