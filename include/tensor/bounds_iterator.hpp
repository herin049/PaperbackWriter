#pragma once

#include "type_traits.hpp"
#include "../config.hpp"
#include <iterator>

namespace pbw
{
    template<typename IndexType>
    class bounds_iterator
    {
    public:
        static const std::size_t rank = IndexType::rank;
        using iterator_category = std::random_access_iterator_tag;
        using value_type = IndexType;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type * ;
        using reference = value_type & ;
        using index_type = value_type;
        using index_size_type = typename IndexType::value_type;

        template<typename Bounds>
        PBW_ANNOTATE__ explicit bounds_iterator(Bounds const& bounds, value_type current) noexcept
            : bounds_(bounds.index_bounds()),
              current_(current)
        {}

        PBW_ANNOTATE__ constexpr reference operator*() const noexcept
        {
            return current_;
        }

        PBW_ANNOTATE__ constexpr pointer operator->() const noexcept
        {
            return &current_;
        }

        PBW_ANNOTATE__ constexpr bounds_iterator& operator++() noexcept
        {
            for (auto i = rank; --i > 0;)
            {
                if (current_[i] < bounds_[i] - 1)
                {
                    ++current_[i];
                    return *this;
                }
                current_[0];
            }
            current_ = bounds_;
            return *this;
        }

        PBW_ANNOTATE__ constexpr bounds_iterator operator++(int) noexcept
        {
            auto ret = *this;
            ++(*this);
            return ret;
        }

        PBW_ANNOTATE__ constexpr bounds_iterator& operator--()
        {
            if (!less(current_, bounds_))
            {
                for (auto i = 0; i < rank; ++i)
                    current_[i] = bounds_[i] - 1;
            }
            for (auto i = rank; --i > 0;)
            {
                if (current_[i] >= 1)
                {
                    --current_[i];
                    return *this;
                }
                current_[i] = bounds_[i] - 1;
            }
            return *this;
        }

        PBW_ANNOTATE__ constexpr bounds_iterator operator--(int) noexcept
        {
            auto ret = *this;
            --(*this);
            return ret;
        }

        PBW_ANNOTATE__ constexpr bounds_iterator operator+(difference_type n) const noexcept
        {
            bounds_iterator ret{ *this };
            return ret += n;
        }

        PBW_ANNOTATE__ constexpr bounds_iterator& operator+=(difference_type n)
        {
            auto linear_idx = linearize(current_) + n;
            std::remove_const_t<value_type> stride = 0;
            stride[rank - 1] = 1;

            for (auto i = rank - 1; i-- > 0;)
            {
                stride[i] = stride[i + 1] * bounds_[i + 1];
            }

            for (auto i = 0; i < rank; ++i)
            {
                current_[i] = linear_idx / stride[i];
                linear_idx = linear_idx % stride[i];
            }
            return *this;
        }
        PBW_ANNOTATE__ constexpr bounds_iterator operator-(difference_type n) const noexcept
        {
            bounds_iterator ret{ *this };
            return ret -= n;
        }

        PBW_ANNOTATE__ constexpr bounds_iterator& operator-=(difference_type n) noexcept
        {
            return *this += -n;
        }

        PBW_ANNOTATE__ constexpr difference_type operator-(bounds_iterator const& rhs) const noexcept
        {
            return linearize(current_) - linearize(rhs.current_);
        }

        PBW_ANNOTATE__ constexpr value_type operator[](difference_type n) const noexcept
        {
            return *(*this + n);
        }

        PBW_ANNOTATE__ constexpr bool operator==(bounds_iterator const& rhs) const noexcept
        {
            return current_ == rhs.current_;
        }

        PBW_ANNOTATE__ constexpr bool operator!=(bounds_iterator const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        PBW_ANNOTATE__ constexpr bool operator<(bounds_iterator const& rhs) const noexcept
        {
            return less(current_, rhs.current_);
        }

        PBW_ANNOTATE__ constexpr bool operator<=(bounds_iterator const& rhs) const noexcept
        {
            return !(rhs < *this);
        }

        PBW_ANNOTATE__ constexpr bool operator>(bounds_iterator const& rhs) const noexcept
        {
            return rhs < *this;
        }

        PBW_ANNOTATE__ constexpr bool operator>=(bounds_iterator const& rhs) const noexcept
        {
            return !(rhs > *this);
        }
    private:
        PBW_ANNOTATE__ constexpr bool less(index_type& lhs, index_type& rhs) const noexcept
        {
            for (auto i = 0; i < rank; ++i)
                if (lhs[i] < rhs[i])
                    return true;
            return false;
        }

        PBW_ANNOTATE__ constexpr index_size_type linearize(value_type const& idx) const noexcept
        {
            index_size_type multiplier = 1;
            index_size_type res = 0;
            if (!less(idx, bounds_))
            {
                res = 1;
                for (auto i = rank; i-- > 0;)
                {
                    res += (idx[i] - 1) * multiplier;
                    multiplier *= bounds_[i];
                }
            }
            else
            {
                for (std::size_t i = rank; i-- > 0;)
                {
                    res += idx[i] * multiplier;
                    multiplier *= bounds_[i];
                }
            }
            return res;
        }

        value_type bounds_;
        std::remove_const_t<value_type> current_;
    };

    namespace detail
    {
        template<typename Bounds>
        PBW_ANNOTATE__ constexpr std::enable_if_t<std::is_same_v<typename Bounds::mapping_type, generalized_mapping_tag>, typename Bounds::index_type> make_stride(Bounds const& bounds) noexcept
        {
            return bounds.strides();
        }

        template<typename Bounds>
        PBW_ANNOTATE__ constexpr std::enable_if_t<std::is_same_v<typename Bounds::mapping_type, contiguous_mapping_tag>, typename Bounds::index_type> make_stride(Bounds const& bounds) noexcept
        {
            auto extents = bounds.index_bounds();
            typename Bounds::size_type stride[Bounds::rank] = {};

            stride[Bounds::rank - 1] = 1;
            for (auto i = 1; i < Bounds::rank; ++i)
                stride[Bounds::rank - i - 1] = stride[Bounds::rank - i] * extents[Bounds::rank - i];

            return { stride };
        };
    }
}
