#pragma once

#include "../../config.hpp"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			__global__ void clip(float* __restrict__ A, const float lower, const float upper, const size_t size);
			__global__ void clip(double* __restrict__ A, const double lower, const double upper, const size_t size);
		}
	}
}
