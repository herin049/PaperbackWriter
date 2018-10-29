#pragma once

#include "clip.cuh"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			__global__ void clip(float* __restrict__ A, const float lower, const float upper, const size_t size)
			{
				const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
				const size_t stride = blockDim.x * gridDim.x;
				for (size_t index = grid; index < size; index += stride)
				{
					const float val_a = A[index];
					const float val_b = fminf(val_a, upper);
					A[index] = fmaxf(val_b, lower);
				}
			}
			__global__ void clip(double* __restrict__ A, const double lower, const double upper, const size_t size)
			{
				const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
				const size_t stride = blockDim.x * gridDim.x;
				for (size_t index = grid; index < size; index += stride)
				{
					const double val_a = A[index];
					const double val_b = fmin(val_a, upper);
					A[index] = fmax(val_b, lower);
				}
			}
		}
	}
}
