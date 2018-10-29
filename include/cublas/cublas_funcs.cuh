#pragma once

#include "cublas_v2.h"
#include "../tensor/tensor_span.hpp"
#include <chrono>
#include "../cuda/kernels/random/sample.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace pbw
{
	namespace cublas
	{
		namespace math
		{
			/* Performs an outer product operation on vectors x, y and addes the result to the matrix A
			A: m x n Matrix
			x: m-Dimensional vector
			y: n-Dimensional vector
			A = x y (T) + A */
			template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB, std::ptrdiff_t... DimsC>
			void outerproduct(cublasHandle_t& handle, pbw::tensor_span<float, DimsA...>& x, pbw::tensor_span<float, DimsB...>& y, pbw::tensor_span<float, DimsC...>& A)
			{
				const float alpha = 1.0f;
				cublasSger(handle, x.size(), y.size(), &alpha, x.data(), 1, y.data(), 1, A.data(), x.size());
			}

			/* Adds the following tensors in an element-wise fashion.
			A: n-Dimensional Tensor
			B: n-Dimensional Tensor
			B += A */
			template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB>
			void add(cublasHandle_t& handle, pbw::tensor_span<float, DimsA...>& A, pbw::tensor_span<float, DimsB...>& B)
			{
				const float alpha = 1.0f;
				cublasSaxpy(handle, A.size(), &alpha, A.data(), 1, B.data(), 1);
			}

			/* Adds the following tensors in an element-wise fashion.
			A: n-Dimensional Tensor
			B: n-Dimensional Tensor
			C: n-Dimensional Tensor
			C = A + B */
			template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB, std::ptrdiff_t... DimsC>
			void add(cublasHandle_t& handle, pbw::tensor_span<float, DimsA...>& A, pbw::tensor_span<float, DimsB...>& B, pbw::tensor_span<float, DimsC...>& C)
			{
				const float alpha = 1.0f;
				cublasScopy(handle, B.size(), B.data(), 1, C.data(), 1);
				cublasSaxpy(handle, A.size(), &alpha, A.data(), 1, C.data(), 1);
			}

			/* Adds the following tensors in an element-wise fashion.
			A: n-Dimensional Tensor
			B: n-Dimensional Tensor
			C: n-Dimensional Tensor
			C = A - B */
			template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB, std::ptrdiff_t... DimsC>
			void subtract(cublasHandle_t& handle, pbw::tensor_span<float, DimsA...>& A, pbw::tensor_span<float, DimsB...>& B, pbw::tensor_span<float, DimsC...>& C)
			{
				const float gamma = -1.0f;
				cublasScopy(handle, A.size(), A.data(), 1, C.data(), 1);
				cublasSaxpy(handle, B.size(), &gamma, B.data(), 1, C.data(), 1);
			}

			//Performs the given matrix multiplication
			// y = Ax 
			// multiplies Matrix A by vector x and asssigns values to y
			template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB, std::ptrdiff_t... DimsC>
			void matrixvectormult(cublasHandle_t& handle, pbw::tensor_span<float, DimsA...>& x, pbw::tensor_span<float, DimsB...>& y, pbw::tensor_span<float, DimsC...>& A)
			{
				const float alpha = 1.0f;
				const float beta = 0.0f;
				cublasSgemv(handle, CUBLAS_OP_N, y.size(), x.size(), &alpha, A.data(), y.size(), x.data(), 1, &beta, y.data(), 1);
			}

			//Performs the given matrix multiplication
			// z = Ax + y
			// multiplies Matrix A by vector x and adds the result to vector y assigning values to y
			template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB, std::ptrdiff_t... DimsC, std::ptrdiff_t... DimsD>
			void matrixvectormult(cublasHandle_t& handle, pbw::tensor_span<float, DimsA...>& x, pbw::tensor_span<float, DimsB...>& y, pbw::tensor_span<float, DimsC...>& z, pbw::tensor_span<float, DimsD...>& A)
			{
				const float alpha = 1.0f;
				cublasScopy(handle, y.size(), y.data(), 1, z.data(), 1);
				cublasSgemv(handle, CUBLAS_OP_N, z.size(), x.size(), &alpha, A.data(), z.size(), x.data(), 1, &alpha, z.data(), 1);
			}

			//Performs the given matrix multiplication
			// y = A(T)x 
			// multiplies Matrix A by vector x and asssigns values to y
			template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB, std::ptrdiff_t... DimsC>
			void matrixvectormultT(cublasHandle_t& handle, pbw::tensor_span<float, DimsA...>& x, pbw::tensor_span<float, DimsB...>& y, pbw::tensor_span<float, DimsC...>& A)
			{
				const float alpha = 1.0f;
				const float beta = 0.0f;
				cublasSgemv(handle, CUBLAS_OP_T, x.size(), y.size(), &alpha, A.data(), x.size(), x.data(), 1, &beta, y.data(), 1);
			}

			//Performs the given matrix multiplication
			// z = A(T)x + y
			// multiplies Matrix A by vector x and adds the result to vector y assigning values to y
			template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB, std::ptrdiff_t... DimsC, std::ptrdiff_t... DimsD>
			void matrixvectormultT(cublasHandle_t& handle, pbw::tensor_span<float, DimsA...>& x, pbw::tensor_span<float, DimsB...>& y, pbw::tensor_span<float, DimsC...>& z, pbw::tensor_span<float, DimsD...>& A)
			{
				const float alpha = 1.0f;
				cublasScopy(handle, y.size(), y.data(), 1, z.data(), 1);
				cublasSgemv(handle, CUBLAS_OP_T, x.size(), z.size(), &alpha, A.data(), x.size(), x.data(), 1, &alpha, y.data(), 1);
			}

			template<std::ptrdiff_t... DimsA>
			void getmax(cublasHandle_t& handle, pbw::tensor_span<float, DimsA...>& x, int& pos)
			{
				cublasIsamax(handle, x.size(), x.data(), 1, &pos);
			}

			template<std::ptrdiff_t... DimsA>
			void sample(cublasHandle_t& handle, pbw::tensor_span<float, DimsA...>& in, int& result)
			{
				int outval;
				unsigned blocks_per_grid = (in.size() + 256 - 1) / 256;
				std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
				auto duration = now.time_since_epoch();
				int seed = (int)(std::chrono::duration_cast<std::chrono::microseconds>(duration).count() % 10000000);
				pbw::cuda::kernels::random::sample<<<blocks_per_grid, 256>>>(in.data(), in.data(), seed, in.size());
				cublasIsamax(handle, in.size(), in.data(), 1, &outval);
				result = outval;
			}

		}
	}
}