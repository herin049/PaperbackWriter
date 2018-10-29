#pragma once

#include "../../../config.hpp"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			namespace activations
			{
				__global__ void relu(const float* __restrict__ A, float* __restrict__ B, size_t size);
				__global__ void relu(const double* __restrict__ A, double* __restrict__ B, size_t size);
				__global__ void drelu(const float* __restrict__ A, float* __restrict__ B, size_t size);
				__global__ void drelu(const double* __restrict__ A, double* __restrict__ B, size_t size);
			}
		}
	}
}