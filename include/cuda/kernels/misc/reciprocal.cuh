#pragma once

#include "../../../config.hpp"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			__global__ void reciprocal(float* __restrict__ A, const float value, const size_t size);
			__global__ void reciprocal(double* __restrict__ A, const double value, const size_t size);
		}
	}
}