#pragma once

#include "tensor_span.hpp"


namespace pbw
{
    template<typename ValueType, std::ptrdiff_t Head, std::ptrdiff_t... Tail>
    class tensor
    {
    public:
        using underlying_type = tensor_span<ValueType, Head, Tail...>;
        using size_type = typename underlying_type::size_type;
        using index_type = typename underlying_type::index_type;
        using pointer = typename underlying_type::pointer;
        using bounds_type = typename underlying_type::bounds_type;
        using iterator = typename underlying_type::pointer;
        using sliced_type = typename underlying_type::sliced_type;

        explicit tensor(size_type size)
            : data_{nullptr}, bounds_{size}
        {
            cudaMalloc((void**)&data_, sizeof(ValueType) * size);
            auto temp_data = new ValueType[bounds_.size()];
            std::fill(&temp_data[0],temp_data + size,0);
            cudaMemcpy(data_, temp_data, sizeof(ValueType) * size, cudaMemcpyHostToDevice);
            delete temp_data;
        }

        operator tensor_span<ValueType,Head,Tail...>() noexcept
        {
            return span();
        }

        operator tensor_span<const ValueType,Head,Tail...>() const noexcept
        {
            return span();
        }

        pointer begin()
        {
            return &data_[0];
        }

        pointer end()
        {
            return data_ + bounds_.size();
        }

        sliced_type operator[](size_type index) noexcept
        {
            return span()[index];
        }

        ~tensor() noexcept
        {
            cudaFree(data_);
        }

		underlying_type span() const noexcept
		{
			return underlying_type{ data_, bounds_ };
		}

    private:
        pointer data_;
        bounds_type bounds_;
    };
}
