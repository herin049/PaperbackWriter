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
				__global__ void tanh_(const float* __restrict__ A, float* __restrict__ B, size_t size);
				__global__ void tanh_(const double* __restrict__ A, double* __restrict__ B, size_t size);
				__global__ void dtanh(const float* __restrict__ A, float* __restrict__ B, size_t size);
				__global__ void dtanh(const double* __restrict__ A, double* __restrict__ B, size_t size);
			}
		}
	}
}