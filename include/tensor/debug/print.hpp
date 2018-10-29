#pragma once

#include "../tensor_span.hpp"

namespace pbw
{
	namespace debug
	{
		//Prints a matrix in column major.
		template<std::ptrdiff_t... DimsA>
		void matrixprint(pbw::tensor_span<float, DimsA...>& matrix, const size_t rows)
		{
			/* Code to copy GPU data into temp matrix to print. */
			float* temp_data = (float *)malloc(sizeof(float) * matrix.size());
			cudaMemcpy(temp_data, matrix.data(), sizeof(float) * matrix.size(), cudaMemcpyDeviceToHost);

			for (int i = 0; i < rows; i++)
			{
				std::cout << "[";
				for (int j = i; j < matrix.size(); j += rows)
				{
					std::cout << temp_data[j] << " ";
				}
				std::cout << "]\n";
			}

			free(temp_data);
		}

		//Prints a vector..
		template<std::ptrdiff_t... DimsA>
		void vectorprint(pbw::tensor_span<float, DimsA...>& vector)
		{
			/* Code to copy GPU data into temp matrix to print. */
			float* temp_data = (float *)malloc(sizeof(float) * vector.size());
			cudaMemcpy(temp_data, vector.data(), sizeof(float) * vector.size(), cudaMemcpyDeviceToHost);

			std::cout << "[";
			for (int i = 0; i < vector.size(); i++)
			{
				std::cout << temp_data[i] << " ";
			}
			std::cout << "]\n";

			free(temp_data);
		}

		template<std::ptrdiff_t... DimsA>
		void quickprint(pbw::tensor_span<float, DimsA...>& a)
		{
			/* Code to copy GPU data into temp matrix to print. */
			float* temp_data = (float *)malloc(sizeof(float) * 100);
			cudaMemcpy(temp_data, a.data(), sizeof(float) * 100, cudaMemcpyDeviceToHost);

			std::cout << "[";
			for (int i = 0; i < 100; i++)
			{
				std::cout << temp_data[i] << " ";
			}
			std::cout << "]\n";

			free(temp_data);
		}
	}
}