#pragma once

#include "exp.cuh"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			__global__ void exp(float* __restrict__ A, float * __restrict__ out, const size_t size)
			{
				const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
				const size_t stride = blockDim.x * gridDim.x;
				for (size_t index = grid; index < size; index += stride)
				{
					float val_a = A[index];
					out[index] = expf(val_a);
				}
			}
			__global__ void exp(double* __restrict__ A, float * __restrict__ out, const size_t size)
			{
				const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
				const size_t stride = blockDim.x * gridDim.x;
				for (size_t index = grid; index < size; index += stride)
				{
					const double val_a = A[index];
					out[index] = ::exp(val_a);
				}
			}
		}
	}
}