#pragma once

#include "tensor_span.hpp"
#include <iterator>
#include <random>


namespace pbw
{
    namespace tensor_funcs
    {
        namespace random
        {
            template<typename ValueType,std::ptrdiff_t Head,std::ptrdiff_t... Tail>
                    void normal(pbw::tensor_span<ValueType, Head, Tail...>& A, const float mean, const float standard_deviation)
            {
                float* temp_data = (float *)malloc(sizeof(float) * A.size());
                for (int i = 0; i < A.size(); i++)
                {
                    static std::random_device Random{};
                    static std::default_random_engine Engine{ Random() };
                    std::normal_distribution<float> Distribution{ mean, standard_deviation };
                    temp_data[i] = Distribution(Engine);
                }
                cudaMemcpy(A.data(), temp_data, sizeof(float) * A.size(), cudaMemcpyHostToDevice);
                free(temp_data);
            }

			template<typename ValueType, std::ptrdiff_t Head, std::ptrdiff_t... Tail>
			void normal(pbw::tensor_span<ValueType, Head, Tail...>& A, const double mean, const double standard_deviation)
			{
				double* temp_data = (double *)malloc(sizeof(double) * A.size());
				for (int i = 0; i < A.size(); i++)
				{
					static std::random_device Random{};
					static std::default_random_engine Engine{ Random() };
					std::normal_distribution<double> Distribution{ mean, standard_deviation };
					temp_data[i] = Distribution(Engine);
				}
				cudaMemcpy(A.data(), temp_data, sizeof(double) * A.size(), cudaMemcpyHostToDevice);
				free(temp_data);
			}
        }
    }
}