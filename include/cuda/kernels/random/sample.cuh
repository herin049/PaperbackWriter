#pragma once

#include "../../../config.hpp"
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			namespace random
			{
				__global__ void sample(const float* __restrict__ A, float* __restrict__ buffer, const int seed, const size_t size);
				__global__ void sample(const double* __restrict__ A, double* __restrict__ buffer, const int seed, const size_t size);
			}
		}
	}
}