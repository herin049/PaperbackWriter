#pragma once

#include "detail/bounds_ranges.hpp"
#include "tensor_index.hpp"

namespace pbw
{
    template<std::size_t Rank>
    class strided_bounds
    {
        template<std::size_t OtherRank>
        friend class strided_bounds;
    public:
        using value_type = std::ptrdiff_t;
        PBW_ANNOTATE__ static std::size_t const rank = Rank;
        static value_type const dynamic_rank = rank;
        static value_type const static_size = dynamic_range;

        using reference = value_type & ;
        using const_reference = value_type const&;
        using size_type = value_type;
        using difference_type = value_type;
        using index_type = tensor_index<rank>;
        using const_index_type = index_type const;
        using iterator = bounds_iterator<const_index_type>;
        using const_iterator = bounds_iterator<const_index_type>;
        using sliced_type = std::conditional_t<rank != 0, strided_bounds<rank - 1>, void>;
        using mapping_type = generalized_mapping_tag;
    public:
        PBW_ANNOTATE__ constexpr strided_bounds(strided_bounds const&) noexcept = default;
        PBW_ANNOTATE__ constexpr strided_bounds& operator=(strided_bounds const&) noexcept = default;

        PBW_ANNOTATE__ constexpr strided_bounds(value_type const(&values)[rank], index_type strides)
            : extents_(values),
              strides_(pbw::move(strides))
        {}

        PBW_ANNOTATE__ constexpr strided_bounds(index_type const& extents, index_type const& strides) noexcept
            : extents_(extents),
              strides_(strides)
        {}

        PBW_ANNOTATE__ constexpr index_type strides() const noexcept
        {
            return strides_;
        }

        PBW_ANNOTATE__ constexpr size_type total_size() const noexcept
        {
            size_type ret = 0;
            for (auto i = 0; i < Rank; ++i)
                ret += (extents_[i] - 1) * strides_[i];
            return ret;
        }

        PBW_ANNOTATE__ constexpr size_type size() const noexcept
        {
            size_type ret = 0;
            for (auto i = 0; i < Rank; ++i)
                ret *= extents_[i];
            return ret;
        }

        PBW_ANNOTATE__ constexpr bool contains(index_type const& index) const noexcept
        {
            for (auto i = 0; i < rank; ++i)
            {
                if (index[i] < 0 || index[i] >= extents_[i])
                    return false;
            }
            return true;
        }

        PBW_ANNOTATE__ constexpr size_type linearize(index_type const& index) const
        {
            size_type ret = 0;
            for (auto i = 0; i < rank; ++i)
                ret += index[i] * strides_[i];
            return ret;
        }

        PBW_ANNOTATE__ constexpr size_type stride() const noexcept
        {
            return strides_[0];
        }

        template<bool Enabled = (rank > 1), typename Return = std::enable_if_t<Enabled, sliced_type>>
        PBW_ANNOTATE__ constexpr Return slice() const
        {
            return { detail::shift_left(extents_),detail::shift_left(strides_) };
        }

        template<std::size_t Dim = 0>
        PBW_ANNOTATE__ constexpr size_type extent() const noexcept
        {
            return extents_[Dim];
        }

        PBW_ANNOTATE__ constexpr index_type index_bounds() const noexcept
        {
            return extents_;
        }

        PBW_ANNOTATE__ constexpr const_iterator begin() const noexcept
        {
            return const_iterator{ *this,index_type{} };
        }

        PBW_ANNOTATE__ constexpr const_iterator end() const noexcept
        {
            return const_iterator{ *this,index_bounds() };
        }
    private:
        PBW_ANNOTATE__ index_type extents_;
        PBW_ANNOTATE__ index_type strides_;
    };

    template<std::size_t Rank>
    struct is_bounds<strided_bounds<Rank>> : std::integral_constant<bool, true>
    {};
}
