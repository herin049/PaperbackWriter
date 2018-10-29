#pragma once

#include "../../../config.hpp"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			namespace optimizers
			{
				__global__ void adagrad(float* __restrict__ weights, float* __restrict__ delta_weights, float* memory, const float learning_rate, const float stabalizer, const size_t size);
				__global__ void adagrad(double* __restrict__ weights, const double* __restrict__ delta_weights, double* memory, const double learning_rate, const double stabalizer, const size_t size);
			}
		}
	}
}