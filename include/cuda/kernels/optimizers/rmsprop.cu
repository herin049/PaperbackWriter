#pragma once

#include "rmsprop.cuh"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			namespace optimizers
			{
				__global__ void rmsprop(float* __restrict__ weights, const float* __restrict__ delta_weights, float* memory, const float learning_rate, const float decay_rate, const float stabalizer, const size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const float delta = delta_weights[index];
						memory[index] = decay_rate * memory[index] + ((1 - decay_rate) * delta * delta);
						weights[index] += (-1 * learning_rate) / (sqrtf(memory[index]) + stabalizer);
					}
				}
				__global__ void rmsprop(double* __restrict__ weights, const double* __restrict__ delta_weights, double* memory, const double learning_rate, const double decay_rate, const double stabilizer, const size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const double delta = delta_weights[index];
						memory[index] = decay_rate * memory[index] + ((1 - decay_rate) * delta * delta);
						weights[index] += (-1 * learning_rate) / (sqrt(memory[index]) + stabilizer);
					}
				}
			}
		}
	}
}