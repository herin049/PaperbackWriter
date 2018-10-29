#pragma once


#include <iostream>
#include <iterator>
#include <algorithm>
#include "type_traits.hpp"
#include "../config.hpp"

namespace pbw
{
    template<std::size_t Rank>
    class tensor_index
    {
        static_assert(Rank > 0, "A tensor index's rank must be greater then 0!");
        template<std::size_t OtherRank>
        friend class tensor_index;
    public:
        static const std::size_t rank = Rank;
        using value_type = std::ptrdiff_t;
        using size_type = value_type;
        using reference = value_type & ;
        using pointer = value_type*;
        using const_reference = value_type const&;

        PBW_ANNOTATE__ constexpr tensor_index() noexcept = default;
        PBW_ANNOTATE__ constexpr tensor_index(tensor_index const& other) noexcept = default;
        PBW_ANNOTATE__ constexpr tensor_index &operator=(tensor_index const &rhs) noexcept = default;

        PBW_ANNOTATE__ constexpr tensor_index(value_type const(&values)[rank]) noexcept
            : elements_{ 0 }
        {
            for (auto i = 0; i < Rank; ++i)
                elements_[i] = values[i];
        }

        template<typename... Ts, class = std::enable_if_t<sizeof...(Ts) == Rank>>
        PBW_ANNOTATE__ constexpr tensor_index(Ts... dims) noexcept
            : elements_{ dims... }
        {}

        PBW_ANNOTATE__ constexpr pointer data()
        {
            return &elements_[0];
        }

        PBW_ANNOTATE__ constexpr reference operator[](std::size_t index)
        {
            return elements_[index];
        }

        PBW_ANNOTATE__ constexpr const_reference operator[](std::size_t index) const
        {
            return elements_[index];
        }

        PBW_ANNOTATE__ constexpr bool operator==(tensor_index const& rhs) const
        {
            for (auto i = 0; i < Rank; ++i)
            {
                if (!(elements_[i] == rhs[i]))
                    return false;
            }
            return true;
        }

        PBW_ANNOTATE__ constexpr bool operator!=(tensor_index const& rhs) const
        {
            return !(this == rhs);
        }
    private:
        value_type elements_[Rank];
    };

    namespace detail
    {
        template<std::size_t Rank, bool Enabled = (Rank > 1), typename Ret = std::enable_if_t<Enabled, tensor_index<Rank - 1>>>
        PBW_ANNOTATE__ constexpr Ret shift_left(tensor_index<Rank> const&other) noexcept
        {
            Ret ret{};
            for (std::size_t i = 0; i < Rank - 1; ++i)
                ret[i] = other[i + 1];
            return ret;
        }
    }
}
