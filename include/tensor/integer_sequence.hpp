#pragma once

#include <cstddef>
#include "type_traits.hpp"

namespace pbw
{
	template<typename T, T... Is>
	struct integer_sequence
	{
		using type = integer_sequence;
		using value_type = T;

		static constexpr std::size_t size() noexcept
		{
			return sizeof...(Is);
		}
	};

	namespace impl
	{
		template<typename, typename>
		struct concatenate;

		template<typename TA, TA... As, typename TB, TB... Bs>
		struct concatenate<integer_sequence<TA, As...>, integer_sequence<TB, Bs...> >
		{
			using type = integer_sequence<TA, As..., Bs...>;
		};

		template<typename T, std::size_t Index, T Value, T... Rest>
		struct remove_at_l
		{
			using super = typename remove_at_l<T, Index - 1, Rest...>::type;
			using type = typename concatenate<integer_sequence<T, Value>, super>::type;
		};

		template<typename T, T V, T... Is>
		struct remove_at_l<T,0, V, Is...>
		{
			using type = integer_sequence<T, Is...>;
		};

		template<typename T, std::size_t Index, typename Sequence>
		struct remove_at;

		template<typename T, std::size_t I, T... Is>
		struct remove_at<T, I, integer_sequence<T, Is...>>
		{
			using type = typename remove_at_l<T, I, Is...>::type;
		};
	}

	template<typename SA,typename SB>
	using concat = typename impl::concatenate<SA,SB>::type;

	template <std::size_t I, typename S>
	using remove_at = typename impl::remove_at<typename S::value_type, I, S>::type;
}
