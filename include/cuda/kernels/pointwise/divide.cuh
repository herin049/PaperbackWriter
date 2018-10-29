#pragma once

#include "../../../config.hpp"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			namespace pointwise
			{
				__global__ void divide(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, const size_t size);
				__global__ void divide(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, const size_t size);
			}
		}
	}
}