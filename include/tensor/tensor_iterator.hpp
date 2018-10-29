#pragma once

#include "../config.hpp"
#include <iostream>

namespace pbw
{
    template <int I, int... N>
    struct tensor_iterator
    {
        template <typename F, typename... X>
        PBW_ANNOTATE__ constexpr void operator()(F& f, X... x)
        {
            for (int i = 0; i < I; ++i)
                tensor_iterator<N...>()(f, x..., i);
        }
    };

    template <int I>
    struct tensor_iterator<I>
    {
        template <typename F, typename... X>
        PBW_ANNOTATE__ constexpr void operator()(F& f, X... x)
        {
            for (int i = 0; i < I; ++i)
                f(x..., i);
        }
    };


    template <int I, int... N>
    struct tensor_print_helper
    {
    public:
        explicit tensor_print_helper(std::ostream& os)
            : ostream_(std::move(os))
        {}

        template <typename F, typename... X>
        void operator()(F& f, X... x)
        {
            for (int i = 0; i < I; ++i)
            {
                ostream_ << '[';
                tensor_print_helper<N...>(std::move(ostream_))(f, x...,i);
                ostream_ << ']';
            }
        }

    private:
        std::ostream& ostream_;
    };

    template <int I>
    struct tensor_print_helper<I>
    {
        explicit tensor_print_helper(std::ostream& os)
            : ostream_(std::move(os))
        {}

        template <typename F, typename... X>
        void operator()(F& f, X... x)
        {
        	char delimiter = "";
            for (int i = 0; i < I; ++i)
            {
                ostream_ << delimiter;
                f(x..., i);
                delimiter = ',';
            }
        }
    private:
        std::ostream& ostream_;
    };
}