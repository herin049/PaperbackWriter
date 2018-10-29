#pragma once

#include "detail/bounds_ranges.hpp"
#include "tensor_index.hpp"
#include "../config.hpp"

namespace pbw
{
    template<std::ptrdiff_t... Ranges>
    class static_bounds
    {
    public:
        PBW_ANNOTATE__ static_bounds(detail::bounds_ranges<Ranges...> const&)
        {}
    };

    template<std::ptrdiff_t FirstRange, std::ptrdiff_t... RestRanges>
    class static_bounds<FirstRange, RestRanges...>
    {
        using bounds_range = detail::bounds_ranges<FirstRange, RestRanges...>;
        template<std::ptrdiff_t... OtherRanges>
        friend class static_bounds;

    public:
        static const std::size_t rank = bounds_range::depth;
        static const std::size_t dynamic_rank = bounds_range::dynamic_num;
        static const std::ptrdiff_t static_size = bounds_range::total_size;

        using size_type = std::ptrdiff_t;
        using index_type = tensor_index<rank>;
        using const_index_type = index_type const;
        using iterator = bounds_iterator<const_index_type>;
        using const_iterator = bounds_iterator<const_index_type>;
        using difference_type = std::ptrdiff_t;
        using sliced_type = static_bounds<RestRanges...>;
        using mapping_type = contiguous_mapping_tag;

        PBW_ANNOTATE__ constexpr static_bounds() /*noexcept*/ = default;
        PBW_ANNOTATE__ constexpr static_bounds(static_bounds const&) noexcept = default;

        template<typename SourceType, typename TargetType, std::size_t Rank>
        struct range_helper;

        template<std::size_t Rank, typename SourceType, typename TargetType, typename Return = range_helper<typename SourceType::base_type, typename TargetType::base_type, Rank>>
         static auto conversion_helper(SourceType, TargetType, std::true_type)
            -> Return;

        template<std::size_t Rank, typename SourceType, typename TargetType>
         static auto conversion_helper(SourceType, TargetType, ...)
            -> std::false_type;

        template <typename SourceType, typename TargetType, std::size_t Rank>
         struct range_helper
            : decltype(conversion_helper<Rank - 1>(SourceType(), TargetType(),
                std::integral_constant<bool,
                SourceType::depth == TargetType::depth && (SourceType::current_range == TargetType::current_range ||
                TargetType::current_range == dynamic_range ||
                SourceType::current_range == dynamic_range)>()))
        {};

        template <typename SourceType, typename TargetType>
         struct range_helper<SourceType, TargetType, 0> : std::true_type
        {};

        template <typename SourceType, typename TargetType, std::ptrdiff_t Rank = TargetType::Depth>
         struct bounds_range_convertible
            : decltype(conversion_helper<Rank - 1>(SourceType(), TargetType(),
                std::integral_constant<bool,
                SourceType::Depth == TargetType::depth &&
                (!less_than<SourceType::current_range, TargetType::current_range>::value ||
                TargetType::current_range == dynamic_range ||
                SourceType::current_range == dynamic_range)>()))
        {};

        template <typename SourceType, typename TargetType>
         struct bounds_range_convertible<SourceType, TargetType, 0> : std::true_type
        {};

        PBW_ANNOTATE__ constexpr static_bounds(bounds_range const& range) noexcept
            : ranges_(range)
        {}

        template<std::ptrdiff_t... Ranges, typename = std::enable_if_t<detail::bounds_range_convertible<detail::bounds_ranges<Ranges...>, detail::bounds_ranges<FirstRange, RestRanges...>>::value>>
        PBW_ANNOTATE__ constexpr static_bounds(static_bounds<Ranges...> const& other)
            : ranges_(other.ranges_)
        {}

        PBW_ANNOTATE__ constexpr static_bounds(std::initializer_list<size_type> li)
            : ranges_(li.begin())
        {}

        PBW_ANNOTATE__ constexpr sliced_type slice() const noexcept
        {
            return sliced_type{ static_cast<detail::bounds_ranges<RestRanges...> const&>(ranges_) };
        }

        PBW_ANNOTATE__ constexpr size_type stride() const noexcept
        {
            return rank > 1 ? slice().size() : 1;
        }

        PBW_ANNOTATE__ constexpr size_type size() const noexcept
        {
            return ranges_.size();
        }

        PBW_ANNOTATE__ constexpr size_type total_size() const noexcept
        {
            return ranges_.size();
        }

        PBW_ANNOTATE__ constexpr size_type linearize(const index_type& idx) const
        {
            return ranges_.linearize(idx);
        }

        PBW_ANNOTATE__ constexpr bool contains(const index_type& idx) const noexcept
        {
            return ranges_.contains(idx) != -1;
        }

        PBW_ANNOTATE__ constexpr size_type operator[](std::size_t idx) const noexcept
        {
            return ranges_.element_count(idx);
        }

        template <std::size_t Dim = 0>
        PBW_ANNOTATE__ constexpr size_type extent() const noexcept
        {
            static_assert(Dim < rank, "dimension should be less than rank (dimension count starts from 0)");
            return detail::make_indexer(ranges_).template get<Dim>().elementNum();
        }

        template <typename IntType>
        PBW_ANNOTATE__ constexpr size_type extent(IntType dim) const
        {
            static_assert(std::is_integral<IntType>::value, "Dimension parameter must be supplied as an integral type.");
            auto real_dim = static_cast<std::size_t>(dim);
            return ranges_.element_count(real_dim);
        }

        PBW_ANNOTATE__ constexpr index_type index_bounds() const noexcept
        {
            size_type extents[rank] = {};
            ranges_.serialize(extents);
            return { extents };
        }

        template <std::ptrdiff_t... Ranges>
        PBW_ANNOTATE__ constexpr bool operator==(const static_bounds<Ranges...>& rhs) const noexcept
        {
            return this->size() == rhs.size();
        }

        template <std::ptrdiff_t... Ranges>
        PBW_ANNOTATE__ constexpr bool operator!=(const static_bounds<Ranges...>& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        PBW_ANNOTATE__ constexpr const_iterator begin() const noexcept
        {
            return const_iterator(*this, index_type{});
        }

        PBW_ANNOTATE__ constexpr const_iterator end() const noexcept
        {
            return const_iterator(*this, this->index_bounds());
        }
    private:
        bounds_range ranges_;
    };

    template<std::ptrdiff_t... Ranges>
    struct is_bounds<static_bounds<Ranges...>> : std::integral_constant<bool, true>
    {};

    namespace detail
    {
        template <typename... Dimensions>
        struct static_as_multi_span_static_bounds_helper
        {
            using type = static_bounds<(Dimensions::value)...>;
        };
    }
}
