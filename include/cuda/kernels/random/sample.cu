#pragma once

#include "sample.cuh"

namespace pbw
{
	namespace cuda
	{
		namespace kernels
		{
			namespace random
			{
				__global__ void sample(const float* __restrict__ A, float* __restrict__ buffer, const int seed, const size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						curandState_t state;
						curand_init(seed + index, 0, 0, &state);
						float u = curand_uniform(&state);
						buffer[index] = powf(u, 1 / A[index]);
					}
				}
				__global__ void sample(const double* __restrict__ A, double* __restrict__ buffer, const int seed, const size_t size)
				{
					const size_t grid = blockDim.x * blockIdx.x + threadIdx.x;
					const size_t stride = blockDim.x * gridDim.x;
					for (size_t index = grid; index < size; index += stride)
					{
						curandState_t state;
						curand_init(seed + index, 0, 0, &state);
						double u = curand_uniform(&state);
						buffer[index] = pow(u, 1 / A[index]);
					}
				}
			}
		}
	}
}