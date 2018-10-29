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
				__global__ void leakyrelu(const float* __restrict__ A, float* __restrict__ B, const float a, size_t size);
				__global__ void leakyrelu(const double* __restrict__ A, double* __restrict__ B, const double a, size_t size);
				__global__ void dleakyrelu(const float* __restrict__ A, float* __restrict__ B, const float a, const size_t size);
				__global__ void dleakyrelu(const double* __restrict__ A, double* __restrict__ B, const double a, const size_t size);
			}
		}
	}
}