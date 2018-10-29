#pragma once

#include "adam.cuh"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			namespace optimizers
			{
				__global__ void adam(float* __restrict__ weights, const float* __restrict__ delta_weights, float* m, float* v, const float beta1, const float beta2, const float learning_rate, const float stabilizer, const size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const float delta = delta_weights[index];
						m[index] = (m[index] * beta1) + ((1 - beta1) * delta);
						v[index] = (v[index] * beta2) + ((1 - beta2) * delta);
						m[index] = m[index] / (1 - beta1);
						v[index] = v[index] / (1 - beta2);
						weights[index] += (-1 * learning_rate * m[index]) / (sqrtf(v[index]) + stabilizer);
					}
				}
				__global__ void adam(double* __restrict__ weights, const double* __restrict__ delta_weights, double* m, double* v, const double beta1, const double beta2, const double learning_rate, const double stabilizer, const size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const double delta = delta_weights[index];
						m[index] = (m[index] * beta1) + ((1 - beta1) * delta);
						v[index] = (v[index] * beta2) + ((1 - beta2) * delta);
						m[index] = m[index] / (1 - beta1);
						v[index] = v[index] / (1 - beta2);
						weights[index] += (-1 * learning_rate * m[index]) / (sqrtf(v[index]) + stabilizer);
					}
				}
			}
		}
	}
}