#pragma once

#include "adagrad.cuh"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			namespace optimizers
			{
				__global__ void adagrad(float* __restrict__ weights, float* __restrict__ delta_weights, float* memory, const float learning_rate, const float stabalizer, const size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const float delta = delta_weights[index];
						memory[index] += (delta * delta);
						weights[index] += (-1 * learning_rate * delta) / (sqrtf(memory[index]) + stabalizer);
						delta_weights[index] = 0;
					}
				}
				__global__ void adagrad(double* __restrict__ weights, const double* __restrict__ delta_weights, double* memory, const double learning_rate, const double stabalizer, const size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						const double delta = delta_weights[index];
						memory[index] += (delta * delta);
						weights[index] += (-1 * learning_rate * delta) / (sqrt(memory[index]) + stabalizer);
					}
				}
			}
		}
	}
}