#pragma once

#include "../../../config.hpp"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			__global__ void exp(float* __restrict__ A, float * __restrict__ out, const size_t size);
			__global__ void exp(double* __restrict__ A, float * __restrict__ out,  const size_t size);
		}
	}
}