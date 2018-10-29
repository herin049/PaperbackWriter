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
				__global__ void adam(float* __restrict__ weights, const float* __restrict__ delta_weights, float* m, float* v, const float beta1, const float beta2, const float learning_rate, const float stabilizer, const size_t size);
				__global__ void adam(double* __restrict__ weights, const double* __restrict__ delta_weights, double* m, double* v, const double beta1, const double beta2, const double learning_rate, const double stabilizer, const size_t size);
			}
		}
	}
}