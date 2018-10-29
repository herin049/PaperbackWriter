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
				__global__ void sigmoid(const float* __restrict__ A, float* __restrict__ B, size_t size);
				__global__ void sigmoid(const double* __restrict__ A, double* __restrict__ B, size_t size);
				__global__ void dsigmoid(const float* __restrict__ A, float* __restrict__ B, size_t size);
				__global__ void dsigmoid(const double* __restrict__ A, double* __restrict__ B, size_t size);
			}
		}
	}
}