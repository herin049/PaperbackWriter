#pragma once

#include "../../tensor/tensor_span.hpp"

#include "../../cuda/kernels/pointwise/add.cuh"
#include "../../cuda/kernels/pointwise/subtract.cuh"
#include "../../cuda/kernels/pointwise/multiply.cuh"
#include "../../cuda/kernels/pointwise/divide.cuh"
#include "../../cuda/kernels/activations/leakyrelu.cuh"
#include "../../cuda/kernels/activations/relu.cuh"
#include "../../cuda/kernels/activations/sigmoid.cuh"
#include "../../cuda/kernels/activations/tanh.cuh"
#include "../../cuda/kernels/optimizers/adagrad.cuh"
#include "../../cuda/kernels/optimizers/adam.cuh"
#include "../../cuda/kernels/optimizers/rmsprop.cuh"
#include "../../cuda/kernels/random/sample.cuh"
#include "../../cuda/kernels/clip.cuh"
#include "../../cuda/kernels/zero.cuh"
#include "../../cuda/kernels/misc/exp.cuh"
#include "../../cuda/kernels/misc/reciprocal.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cublas_v2.h"

namespace pbw
{
	namespace cuda
	{
		namespace math
		{
			constexpr const unsigned THREADS_PER_BLOCK = 256;
			namespace pointwise
			{
				/* C = A + B
				A: n-Dimensional Tensor
				B: n-Dimensional Tensor
				C: n-Dimensional Tensor */
				template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB, std::ptrdiff_t... DimsC>
				void add(pbw::tensor_span<float, DimsA...>& A, pbw::tensor_span<float, DimsB...>& B, pbw::tensor_span<float, DimsC...>& C)
				{
					unsigned blocks_per_grid = (A.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
					pbw::cuda::kernels::pointwise::add<<<blocks_per_grid, THREADS_PER_BLOCK>>>(A.data(), B.data(), C.data(), A.size());
				}
				/* C = A - B
				A: n-Dimensional Tensor
				B: n-Dimensional Tensor
				C: n-Dimensional Tensor */
				template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB, std::ptrdiff_t... DimsC>
				void subtract(pbw::tensor_span<float, DimsA...>& A, pbw::tensor_span<float, DimsB...>& B, pbw::tensor_span<float, DimsC...>& C)
				{
					unsigned blocks_per_grid = (A.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
					pbw::cuda::kernels::pointwise::subtract<<<blocks_per_grid, THREADS_PER_BLOCK>>>(A.data(), B.data(), C.data(), A.size());
				}
				/* C = A * B
				A: n-Dimensional Tensor
				B: n-Dimensional Tensor
				C: n-Dimensional Tensor */
				template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB, std::ptrdiff_t... DimsC>
					void multiply(pbw::tensor_span<float, DimsA...>& A, pbw::tensor_span<float, DimsB...>& B, pbw::tensor_span<float, DimsC...>& C)
				{
						unsigned blocks_per_grid = (A.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
						pbw::cuda::kernels::pointwise::multiply<<<blocks_per_grid, THREADS_PER_BLOCK>>>(A.data(), B.data(), C.data(), A.size());
				}
				/* C = A / B
				A: n-Dimensional Tensor
				B: n-Dimensional Tensor
				C: n-Dimensional Tensor */
				template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB, std::ptrdiff_t... DimsC>
				void divide(pbw::tensor_span<float, DimsA...>& A, pbw::tensor_span<float, DimsB...>& B, pbw::tensor_span<float, DimsC...>& C)
				{
					unsigned blocks_per_grid = (A.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
					pbw::cuda::kernels::pointwise::divide<<<blocks_per_grid, THREADS_PER_BLOCK >>>(A.data(), B.data(), C.data(), A.size());
				}
			}

			namespace activations
			{
				/* Leaky RELU Activation with slope of "a" from [-inf, 0)
				B = leakyrelu(A)
				A: n-Dimensional Tensor
				B: n-Dimensional Tensor
				*/
				template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB>
				void leakyrelu(pbw::tensor_span<float, DimsA...>& A, pbw::tensor_span<float, DimsB...>& B, const float a)
				{
					unsigned blocks_per_grid = (A.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
					pbw::cuda::kernels::activations::leakyrelu<<<blocks_per_grid, THREADS_PER_BLOCK>>>(A.data(), B.data(), a, A.size());
				}

				/* Leaky RELU Activation derivative with slope of "a" from [-inf, 0)
				B = leakyrelu(A)
				A: n-Dimensional Tensor
				B: n-Dimensional Tensor
				*/
				template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB>
				void dleakyrelu(pbw::tensor_span<float, DimsA...>& A, pbw::tensor_span<float, DimsB...>& B, const float a)
				{
					unsigned blocks_per_grid = (A.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
					pbw::cuda::kernels::activations::dleakyrelu<<<blocks_per_grid, THREADS_PER_BLOCK>>>(A.data(), B.data(), a, A.size());
				}

				/* RELU Activation
				B = relu(A)
				A: n-Dimensional Tensor
				B: n-Dimensional Tensor
				*/
				template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB>
				void relu(pbw::tensor_span<float, DimsA...>& A, pbw::tensor_span<float, DimsB...>& B)
				{
					unsigned blocks_per_grid = (A.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
					pbw::cuda::kernels::activations::relu <<<blocks_per_grid, THREADS_PER_BLOCK >>>(A.data(), B.data(), A.size());
				}

				/* RELU Activation derivative
				B = drelu(A)
				A: n-Dimensional Tensor
				B: n-Dimensional Tensor
				*/
				template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB>
				void drelu(pbw::tensor_span<float, DimsA...>& A, pbw::tensor_span<float, DimsB...>& B)
				{
					unsigned blocks_per_grid = (A.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
					pbw::cuda::kernels::activations::drelu<<<blocks_per_grid, THREADS_PER_BLOCK>>>(A.data(), B.data(), A.size());
				}

				/* Sigmoid Activation
				B = sigmoid(A)
				A: n-Dimensional Tensor
				B: n-Dimensional Tensor
				*/
				template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB>
				void sigmoid(pbw::tensor_span<float, DimsA...>& A, pbw::tensor_span<float, DimsB...>& B)
				{
					unsigned blocks_per_grid = (A.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
					pbw::cuda::kernels::activations::sigmoid<<<blocks_per_grid, THREADS_PER_BLOCK>>>(A.data(), B.data(), A.size());
				}

				/* Sigmoid Activation derivative
				B = dsigmoid(A)
				A: n-Dimensional Tensor
				B: n-Dimensional Tensor
				*/
				template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB>
				void dsigmoid(pbw::tensor_span<float, DimsA...>& A, pbw::tensor_span<float, DimsB...>& B)
				{
					unsigned blocks_per_grid = (A.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
					pbw::cuda::kernels::activations::dsigmoid<<<blocks_per_grid, THREADS_PER_BLOCK>>>(A.data(), B.data(), A.size());
				}

				/* Hyperbolic Tangent Activation
				B = tanh(A)
				A: n-Dimensional Tensor
				B: n-Dimensional Tensor
				*/
				template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB>
				void tanh(pbw::tensor_span<float, DimsA...>& A, pbw::tensor_span<float, DimsB...>& B)
				{
					unsigned blocks_per_grid = (A.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
					pbw::cuda::kernels::activations::tanh_<<<blocks_per_grid, THREADS_PER_BLOCK>>>(A.data(), B.data(), A.size());
				}

				/* Hyperbolic Tangent Activation derivative
				B = dtanh(A)
				A: n-Dimensional Tensor
				B: n-Dimensional Tensor
				*/
				template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB>
				void dtanh(pbw::tensor_span<float, DimsA...>& A, pbw::tensor_span<float, DimsB...>& B)
				{
					unsigned blocks_per_grid = (A.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
					pbw::cuda::kernels::activations::dtanh<<<blocks_per_grid, THREADS_PER_BLOCK>>>(A.data(), B.data(), A.size());
				}


				template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB>
				void softmax(cublasHandle_t& handle, pbw::tensor_span<float, DimsA...>& in, pbw::tensor_span<float, DimsB...>& out)
				{
					float total;
					unsigned blocks_per_grid = (in.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
					pbw::cuda::kernels::exp<<<blocks_per_grid, THREADS_PER_BLOCK>>>(in.data(), out.data(), in.size());
					cublasSasum(handle, in.size(), out.data(), 1, &total);
					pbw::cuda::kernels::reciprocal<<<blocks_per_grid, THREADS_PER_BLOCK>>>(out.data(), total, out.size());
				}
			}

			namespace optimizers
			{
				/* 
				Performes the following "adagrad" gradient descent algorithm with the following parameters. 
				*/
				template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB, std::ptrdiff_t... DimsC>
				void adagrad(pbw::tensor_span<float, DimsA...>& weights, pbw::tensor_span<float, DimsB...>& delta_weights, pbw::tensor_span<float, DimsC...>& memory, const float learning_rate, const float stabalizer)
				{
					unsigned blocks_per_grid = (weights.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
					pbw::cuda::kernels::optimizers::adagrad<<<blocks_per_grid, THREADS_PER_BLOCK>>>(weights.data(), delta_weights.data(), memory.data(), learning_rate, stabalizer, weights.size());
				}

				/*
				Performes the following "adam" gradient descent algorithm with the following parameters.
				*/
				template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB, std::ptrdiff_t... DimsC, std::ptrdiff_t... DimsD>
				void adam(pbw::tensor_span<float, DimsA...>& weights, pbw::tensor_span<float, DimsB...>& delta_weights, pbw::tensor_span<float, DimsC...>& m, pbw::tensor_span<float, DimsD...>& v, const float beta1, const float beta2, const float learning_rate, const float stabilizer)
				{
					unsigned blocks_per_grid = (weights.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
					pbw::cuda::kernels::optimizers::adam<<<blocks_per_grid, THREADS_PER_BLOCK>>>(weights.data(), delta_weights.data(), m.data(), v.data(), beta1, beta2, learning_rate, stabilizer, weights.size());
				}

				/*
				Performes the following "rmsprop" gradient descent algorithm with the following parameters.
				*/
				template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB, std::ptrdiff_t... DimsC>
				void rmsprop(pbw::tensor_span<float, DimsA...>& weights, pbw::tensor_span<float, DimsB...>& delta_weights, pbw::tensor_span<float, DimsC...>& memory, const float learning_rate, const float decay_rate, const float stabalizer)
				{
					unsigned blocks_per_grid = (weights.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
					pbw::cuda::kernels::optimizers::rmsprop<<<blocks_per_grid, THREADS_PER_BLOCK>>>(weights.data(), delta_weights.data(), memory.data(), learning_rate, decay_rate, stabalizer, weights.size());
				}
			}

			namespace random
			{

			}
			/*
			Clips the following n-Dimensional tensor on the range [lower, uppper]
			*/
			template<std::ptrdiff_t... DimsA>
			void clip(pbw::tensor_span<float, DimsA...>& A, const float lower, const float upper)
			{
				unsigned blocks_per_grid = (A.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
				pbw::cuda::kernels::clip<<<blocks_per_grid, THREADS_PER_BLOCK>>>(A.data(), lower, upper, A.size());
			}

			/*
			Zeroes the following n-Dimensional tensor on the range [lower, uppper]
			*/
			template<std::ptrdiff_t... DimsA>
			void zero(pbw::tensor_span<float, DimsA...>& A)
			{
				unsigned blocks_per_grid = (A.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
				pbw::cuda::kernels::zero<<<blocks_per_grid, THREADS_PER_BLOCK>>>(A.data(), A.size());
			}

		}

	}

}