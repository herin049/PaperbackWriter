#pragma once

#include "zero.cuh"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			__global__ void zero(float* __restrict__ A, const size_t size)
			{
				const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
				const size_t stride = blockDim.x * gridDim.x;
				for (size_t index = grid; index < size; index += stride)
				{
					A[index] = 0;
				}
			}
			__global__ void zero(double* __restrict__ A, const size_t size)
			{
				const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
				const size_t stride = blockDim.x * gridDim.x;
				for (size_t index = grid; index < size; index += stride)
				{
					A[index] = 0;
				}
			}
		}
	}
}