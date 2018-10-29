#pragma once

#include "leakyrelu.cuh"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			namespace activations
			{
				__global__ void leakyrelu(const float* __restrict__ A, float* __restrict__ B, const float a, size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const float val_a = A[index];
						B[index] = fmaxf(a * val_a, val_a);
					}
				}
				__global__ void leakyrelu(const double* __restrict__ A, double* __restrict__ B, const double a, size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const double val_a = A[index];
						B[index] = fmax(a * val_a, val_a);
					}
				}
				__global__ void dleakyrelu(const float* __restrict__ A, float* __restrict__ B, const float a, const size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const float val_a = A[index];
						B[index] = (val_a > 0) ? 1 : a;
					}
				}
				__global__ void dleakyrelu(const double* __restrict__ A, double* __restrict__ B, const double a, const size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const double val_a = A[index];
						B[index] = (val_a > 0) ? 1 : a;
					}
				}
			}
		}
	}
}