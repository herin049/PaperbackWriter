#pragma once

#include "tensor_index.hpp"
#include "static_bounds.hpp"
#include "type_traits.hpp"
#include "../config.hpp"
#include <ostream>

namespace pbw
{
    template<typename ValueType, std::ptrdiff_t Head, std::ptrdiff_t... Tail>
    class tensor_span
    {
        template<typename ValueTypeOther, std::ptrdiff_t HeadOther, std::ptrdiff_t... TailOther>
        friend class tensor_span;
    public:
        using bounds_type = static_bounds<Head, Tail...>;
        static std::size_t const rank = bounds_type::rank;
        using size_type = typename bounds_type::size_type;
        using index_type = typename bounds_type::index_type;
        using value_type = ValueType;
        using const_value_type = value_type const;
        using pointer = value_type * ;
        using reference = value_type & ;
        using iterator = contiguous_span_iterator<tensor_span>;
        using const_span = tensor_span<const_value_type, Head, Tail...>;
        using const_iterator = contiguous_span_iterator<const_span>;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        using sliced_type = std::conditional_t<rank == 1, value_type, tensor_span<value_type, Tail...>>;

        constexpr tensor_span& operator=(const tensor_span&) = default;
        constexpr tensor_span& operator=(tensor_span&&) noexcept = default;
        constexpr tensor_span(const tensor_span&) = default;
        constexpr tensor_span(tensor_span&&) noexcept = default;
        constexpr tensor_span(value_type&&) = delete;

        PBW_ANNOTATE__ constexpr tensor_span() noexcept
            : tensor_span(nullptr, bounds_type{})
        {
            static_assert(bounds_type::dynamic_rank != 0 || (bounds_type::dynamic_rank && bounds_type::static_size),
                "Default construction of tensor<T> only possible for dynamic or fixed, zero-length spans.");
        }

        PBW_ANNOTATE__ constexpr explicit tensor_span(std::nullptr_t) noexcept
            : tensor_span(nullptr, bounds_type{})
        {
            static_assert(bounds_type::dynamic_rank != 0 || (bounds_type::dynamic_rank == 0 && bounds_type::static_size == 0),
                "Default construction of tensor<T> only possible for dynamic or fixed, zero-length spans.");
        }

        template <class IntType, typename = std::enable_if_t<std::is_integral<IntType>::value>>
        PBW_ANNOTATE__ constexpr tensor_span(std::nullptr_t, IntType size)
            : tensor_span(nullptr, bounds_type{})
        {
            static_assert(bounds_type::dynamic_rank != 0 || (bounds_type::dynamic_rank == 0 && bounds_type::static_size == 0),
                "Default construction of tensor<T> only possible for dynamic or fixed, zero-length spans.");
        }

        PBW_ANNOTATE__ constexpr tensor_span(pointer ptr, size_type size)
            : tensor_span(ptr, bounds_type{ size })
        {}

        PBW_ANNOTATE__ constexpr tensor_span(pointer data, bounds_type bounds)
            : data_(data), bounds_(std::move(bounds))
        {}

        template <typename Ptr, typename = std::enable_if_t<std::is_convertible<Ptr, pointer>::value != 0>>
        PBW_ANNOTATE__ constexpr tensor_span(pointer begin, Ptr end)
            : tensor_span(begin, detail::new_bounds_helper<bounds_type>(static_cast<pointer>(end) - begin))
        {}

        template <typename T, std::size_t N, typename Helper = detail::tensor_array_traits<T, N>>
        PBW_ANNOTATE__ constexpr tensor_span(T(&arr)[N])
            : tensor_span(reinterpret_cast<pointer>(arr), bounds_type{ typename Helper::bounds_type{} })
        {
            static_assert(std::is_convertible<typename Helper::value_type(*)[], value_type(*)[]>::value, "Cannot convert from source type to target tensor_span type.");
            static_assert(std::is_convertible<typename Helper::bounds_type, bounds_type>::value, "Cannot construct a tensor_span from an array with fewer elements.");
        }

        template <typename T, typename Helper = detail::tensor_array_traits<T, dynamic_range>>
        PBW_ANNOTATE__ constexpr tensor_span(T* const& data, size_type size)
            : tensor_span(reinterpret_cast<pointer>(data), typename Helper::bounds_type{ size })
        {
            static_assert(std::is_convertible<typename Helper::value_type(*)[], value_type(*)[]>::value, "Cannot convert from source type to target multi_span type.");
        }


        template <typename OtherValueType, std::ptrdiff_t... OtherDimensions, typename OtherBounds = static_bounds<OtherDimensions...>, typename = std::enable_if_t<std::is_convertible<OtherValueType, ValueType>::value && std::is_convertible<OtherBounds, bounds_type>::value>>
        PBW_ANNOTATE__ constexpr tensor_span(tensor_span<OtherValueType, OtherDimensions...> other)
            : data_(other.data_), bounds_(other.bounds_)
        {}


        template <typename Cont, typename DataType = typename Cont::value_type,
                typename = std::enable_if_t<!detail::is_tensor_span<Cont>::value &&
                std::is_convertible<DataType(*)[], value_type(*)[]>::value &&
                std::is_same<std::decay_t<decltype(std::declval<Cont>().size(), *std::declval<Cont>().data())>, DataType>::value>>
        PBW_ANNOTATE__ constexpr tensor_span(Cont& cont)
            : tensor_span(static_cast<pointer>(cont.data()), detail::new_bounds_helper<bounds_type>(reinterpret_cast<size_type>(cont.size())))
        {}

        template <typename Cont, typename DataType = typename Cont::value_type,
                typename = std::enable_if_t<!detail::is_tensor_span<Cont>::value &&
                std::is_convertible<DataType(*)[], value_type(*)[]>::value &&
                std::is_same<std::decay_t<decltype(std::declval<Cont>().size(), *std::declval<Cont>().data())>, DataType>::value>>
        PBW_ANNOTATE__ constexpr tensor_span(Cont&& cont) = delete;

        template<std::ptrdiff_t Count>
        PBW_ANNOTATE__ constexpr tensor_span<ValueType, Count> first() const
        {
            return { data(),Count };
        }

        PBW_ANNOTATE__ constexpr tensor_span<ValueType, dynamic_range> first(size_type count) const
        {
            return { data(),count };
        }

        template<std::ptrdiff_t Count>
        PBW_ANNOTATE__ constexpr tensor_span<ValueType, Count> last() const
        {
            return { data() + size() - Count,Count };
        }

        PBW_ANNOTATE__ constexpr tensor_span<ValueType, dynamic_range> last(size_type count) const
        {
            return  { data() + size() - count,count };
        }

        PBW_ANNOTATE__ constexpr size_type size() const noexcept
        {
            return bounds_.size();
        }

        PBW_ANNOTATE__ constexpr size_type length() const noexcept
        {
            return this->size();
        }

        PBW_ANNOTATE__ constexpr size_type size_bytes() const noexcept
        {
            return sizeof(value_type) * this->size();
        }

        PBW_ANNOTATE__ constexpr size_type length_bytes() const noexcept
        {
            return this->size_bytes();
        }

        PBW_ANNOTATE__ constexpr bool empty() const noexcept
        {
            return this->size() == 0;
        }

        PBW_ANNOTATE__ constexpr pointer data() const noexcept
        {
            return data_;
        }

        PBW_ANNOTATE__ constexpr bounds_type bounds() const noexcept
        {
            return bounds_;
        }

        template <std::size_t Dim = 0>
        PBW_ANNOTATE__ constexpr size_type extent() const noexcept
        {
            static_assert(Dim < rank, "Dimension should be less than rank (dimension count starts from 0).");
            return bounds_.template extent<Dim>();
        }

        template <typename IntType>
        PBW_ANNOTATE__ constexpr size_type extent(IntType dim) const
        {
            return bounds_.extent(dim);
        }

        template<typename FirstIndex>
        PBW_ANNOTATE__ constexpr reference operator()(FirstIndex idx) const noexcept
        {
            return this->operator[](idx);
        }

        template<typename FirstIndex, typename... OtherIndices>
        PBW_ANNOTATE__ constexpr reference operator()(FirstIndex first, OtherIndices... indices) const noexcept
        {
            index_type const idx = { first,(indices)... };
            return this->operator[](idx);
        }

        PBW_ANNOTATE__ constexpr reference operator[](const index_type& idx) const
        {
            return data_[bounds_.linearize(idx)];
        }

        template <bool Enabled = (rank > 1), typename Ret = std::enable_if_t<Enabled, sliced_type>>
        PBW_ANNOTATE__ constexpr Ret operator[](size_type idx) const
        {
            size_type const index = idx * bounds_.stride();
            return Ret{ data_ + index, bounds_.slice() };
        }

        PBW_ANNOTATE__ constexpr iterator begin() const noexcept
        {
            return iterator{ this, true };
        }

        PBW_ANNOTATE__ constexpr iterator end() const noexcept
        {
            return iterator{ this, false };
        }

        friend std::ostream &operator<<(std::ostream &os, const tensor_span<ValueType,Head,Tail...>& span)
        {

            float* temp_data = (float *)malloc(sizeof(ValueType) * span.size());
            cudaMemcpy(temp_data, span.data(), sizeof(ValueType) * span.size(), cudaMemcpyDeviceToHost);

            for (int j = 0; j < span.rank; j++)
            {
                os << "[";
            }

            for (int i = 0; i < span.size(); i++)
            {
                os << temp_data[i];

                unsigned int brackets = 0;
                unsigned int total = 1;
                for (size_t k = span.rank - 1; k > 0; k--)
                {
                    total *= span.extent(k);

                    if (((i + 1) % total) == 0)
                    {
                        os << "]";
                        brackets++;
                    }

                }
                if (brackets != 0 && i != span.size() - 1)
                {
                    os << "\n";
                    for (int p = 0; p < brackets; p++)
                    {
                        os << "[";
                    }
                }
                if(brackets == 0 && i != span.size() - 1)
                {
                    os << ", ";
                }
            }
            os << "]";
            free(temp_data);
            return os;
        }

    private:
        pointer data_;
        bounds_type bounds_;
        friend iterator;
        friend const_iterator;
    };

}
