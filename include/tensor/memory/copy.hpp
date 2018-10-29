#pragma once

#include "../tensor_span.hpp"

namespace pbw
{
	namespace tensor_funcs
	{
		template<std::ptrdiff_t... DimsA>
		void copy(const float* source, pbw::tensor_span<float, DimsA...>& destination)
		{
			cudaMemcpy(destination.data(), source, destination.size() * sizeof(float), cudaMemcpyHostToDevice);
		}

		template<std::ptrdiff_t... DimsA>
		void copy(pbw::tensor_span<float, DimsA...>& source, float * destination)
		{
			cudaMemcpy(destination, source.data(), source.size() * sizeof(float), cudaMemcpyDeviceToHost);
		}

		template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB>
		void copy(pbw::tensor_span<float, DimsA...>& source, pbw::tensor_span<float, DimsB...>& destination)
		{
			cudaMemcpy(destination.data(), source.data(), source.size() * sizeof(float), cudaMemcpyDeviceToDevice);
		}

		template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB, std::ptrdiff_t... DimsC>
		void concatinate(pbw::tensor_span<float, DimsA...>& tensor_A, pbw::tensor_span<float, DimsB...>& tensor_B, pbw::tensor_span<float, DimsC...>& destination)
		{
			cudaMemcpy(destination.data(), tensor_A.data(), sizeof(float) * tensor_A.size(), cudaMemcpyDeviceToDevice);
			cudaMemcpy(destination.data() + tensor_A.size(), tensor_B.data(), tensor_B.size() * sizeof(float), cudaMemcpyDeviceToDevice);
		}

		template<std::ptrdiff_t... DimsA, std::ptrdiff_t... DimsB>
		void subset(pbw::tensor_span<float, DimsA...>& source, pbw::tensor_span<float, DimsB...>& destination, const size_t offset)
		{
			cudaMemcpy(destination.data(), source.data(), (source.size() - offset) * sizeof(float), cudaMemcpyDeviceToDevice);
		}

	}
}