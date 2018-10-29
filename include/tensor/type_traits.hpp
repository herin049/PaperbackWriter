#pragma once

#include <type_traits>

namespace pbw
{
    const std::ptrdiff_t dynamic_range = -1;

    template <std::ptrdiff_t DimValue = dynamic_range>
    struct dim_t
    {
        static const std::ptrdiff_t value = DimValue;
    };

    template <>
    struct dim_t<dynamic_range>
    {

        constexpr explicit dim_t(std::ptrdiff_t const size) noexcept
                : dim_value(size) {}

        static const std::ptrdiff_t value = dynamic_range;
        std::ptrdiff_t const dim_value;
    };

    template <std::ptrdiff_t N, class = std::enable_if_t<(N >= 0)>>
    constexpr dim_t<N> dim() noexcept
    {
        return dim_t<N>();
    }

    template <std::ptrdiff_t N = dynamic_range, class = std::enable_if_t<N == dynamic_range>>
    constexpr dim_t<N> dim(std::ptrdiff_t const n) noexcept
    {
        return dim_t<>(n);
    }


    template <typename... Ts>
    class are_integral : public std::integral_constant<bool, true> {};

    template <typename T, typename... Ts>
    class are_integral<T, Ts...> : public std::integral_constant<bool, std::is_integral<T>::value != 0> {};

    struct generalized_mapping_tag {};
    struct contiguous_mapping_tag : generalized_mapping_tag {};

    template<std::ptrdiff_t Left, std::ptrdiff_t Right>
    struct less_than
    {
        static const bool value = Left < Right;
    };

    template<std::ptrdiff_t Left, std::ptrdiff_t Right>
    auto less_than_v = less_than<Left, Right>::value;


    template<typename IndexType>
    class bounds_iterator;

    template<typename _Tp>
    constexpr typename std::remove_reference<_Tp>::type&& move(_Tp&& __t) noexcept
    {
        return __t;
    }

    template <typename Span>
    class contiguous_span_iterator;

    template <typename Span>
    class general_span_iterator;


    template<typename T>
    struct is_bounds : std::integral_constant<bool, false>
    {};


    template <typename ValueType, std::ptrdiff_t Head = dynamic_range, std::ptrdiff_t... Tail>
    class tensor_span;

    template <typename ValueType, std::size_t Rank>
    class strided_span;


    namespace detail
    {
        template<typename T, typename = std::true_type>
        struct tensor_type_traits {
            using value_type = T;
            using size_type = std::size_t;
        };

        template<typename Traits>
        struct tensor_type_traits<Traits, typename std::is_reference<typename Traits::tensor_traits &>::type> {
            using value_type = typename Traits::tensor_traits::value_type;
            using size_type = typename Traits::tensor_traits::size_type;
        };

        template<typename T, std::ptrdiff_t... Ranks>
        struct tensor_array_traits {
            using type = tensor_span<T, Ranks...>;
            using value_type = T;
            //using bounds_type = static_bounds<Ranks...>;
            using pointer = T * ;
            using reference = T & ;
        };

        template<typename T, std::ptrdiff_t N, std::ptrdiff_t... Ranks>
        struct tensor_array_traits<T[N], Ranks...> : tensor_array_traits<T, Ranks..., N>
        {

        };

        template<typename BoundsType>
        BoundsType new_bounds_helper_impl(std::ptrdiff_t total_size, std::true_type) {
            return BoundsType{ total_size };
        }

        template<typename BoundsType>
        BoundsType new_bounds_helper_impl(std::ptrdiff_t total_size, std::false_type) {
            return {};
        }

        template<typename BoundsType>
        BoundsType new_bounds_helper(std::ptrdiff_t total_size)
        {
            return new_bounds_helper_impl<BoundsType>(total_size, std::integral_constant<bool, BoundsType::dynamic_rank == 1>());
        }

        struct seperator {};

        template <typename T, typename... Args>
        T static_as_tensor_span_helper(seperator, Args... args)
        {
            return T{ static_cast<typename T::size_type>(args)... };
        }

        template <typename T, typename Arg, typename... Args>
        std::enable_if_t<!std::is_same<Arg, dim_t<dynamic_range>>::value && !std::is_same<Arg, seperator>::value, T> static_as_multi_span_helper(Arg, Args... args)
        {
            return static_as_tensor_span_helper<T>(args...);
        }

        template <typename T, typename... Args>
        T static_as_tensor_span_helper(dim_t<dynamic_range> val, Args... args)
        {
            return static_as_multi_span_helper<T>(args..., val.dim_value);
        }

        template <typename T>
        struct is_tensor_span_oracle : std::false_type
        {};

        template <typename ValueType, std::ptrdiff_t FirstDimension, std::ptrdiff_t... RestDimensions>
        struct is_tensor_span_oracle<tensor_span<ValueType, FirstDimension, RestDimensions...>> : std::true_type
        {};

        template <typename ValueType, std::ptrdiff_t Rank>
        struct is_tensor_span_oracle<strided_span<ValueType, Rank>> : std::true_type
        {};

        template <typename T>
        struct is_tensor_span : is_tensor_span_oracle<std::remove_cv_t<T>>
        {};
    }
}
