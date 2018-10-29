#pragma once

#include "../../config.hpp"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			__global__ void zero(float* __restrict__ A, const size_t size);
			__global__ void zero(double* __restrict__ A, const size_t size);
		}
	}
}