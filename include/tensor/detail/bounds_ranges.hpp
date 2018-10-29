#pragma once

#include "../type_traits.hpp"
#include "../../config.hpp"


namespace pbw
{
    namespace detail
    {
        template<std::ptrdiff_t... Ranges>
        struct bounds_ranges
        {
            using size_type = std::ptrdiff_t;
            static const size_type depth = 0;
            static const size_type dynamic_num = 0;
            static const size_type current_range = 1;
            static const size_type total_size = 1;

            PBW_ANNOTATE__ bounds_ranges() noexcept = default;

            PBW_ANNOTATE__ bounds_ranges(std::ptrdiff_t const*) {}

            template<typename OtherRange>
            PBW_ANNOTATE__ bounds_ranges(OtherRange const &, bool) {}

            template<typename T, std::size_t Dim>
            PBW_ANNOTATE__ void serialize(T&) const {}

            template<typename T, std::size_t Dim>
            size_type linearize(T const &) const
            {
                return 0;
            }

            template<typename T, std::size_t Dim>
            PBW_ANNOTATE__ size_type contains(T const &) const
            {
                return -1;
            }

            PBW_ANNOTATE__ size_type element_count(std::size_t) const noexcept
            {
                return 0;
            }

            PBW_ANNOTATE__ size_type size() const noexcept
            {
                return total_size;
            }

            PBW_ANNOTATE__ bool operator==(bounds_ranges const &) const noexcept
            {
                return true;
            }
        };

        template<std::ptrdiff_t... RestRanges>
        struct bounds_ranges<dynamic_range, RestRanges...> : bounds_ranges<RestRanges...> {
            using base_type = bounds_ranges<RestRanges...>;
            using size_type = std::ptrdiff_t;
            static const size_type depth = 0;
            static const size_type dynamic_num = 0;
            static const size_type current_range = 1;
            static const size_type total_size = 1;

            PBW_ANNOTATE__ bounds_ranges() noexcept
                    : bounds_{ 0 } {}

            PBW_ANNOTATE__ bounds_ranges(std::ptrdiff_t const *array)
                    : base_type(array + 1),
                      bounds_(*array + this->base_type::size()) {}

            template<std::ptrdiff_t OtherRange, std::ptrdiff_t... RestOtherRanges>
            PBW_ANNOTATE__ bounds_ranges(bounds_ranges<OtherRange, RestOtherRanges...> const &other, bool = true)
                    : base_type(other, false),
                      bounds_(other.size()) {}

            template<typename T, std::size_t Dim = 0>
            PBW_ANNOTATE__ void serialize(T& array) const
            {
                array[Dim] = element_count();
                this->base_type::template serialize<T, Dim + 1>(array);
            }

            template<typename T, std::size_t Dim = 0>
            PBW_ANNOTATE__ size_type linearize(T const &array) const
            {
                size_type const index = this->base_type::size() * array[Dim];
                return index + this->base_type::template linearize<T, Dim + 1>(array);
            }

            template<typename T, std::size_t Dim = 0>
            PBW_ANNOTATE__ size_type contains(T const &array) const
            {
                const ptrdiff_t last = this->base_type::template contains<T, Dim + 1>(array);
                if (last == -1) return -1;
                const ptrdiff_t cur = this->base_type::size() * array[Dim];
                return cur < bounds_ ? cur + last : -1;
            }

            PBW_ANNOTATE__ size_type element_count() const noexcept
            {
                return size() / this->base_type::size();
            }

            PBW_ANNOTATE__ size_type element_count(std::size_t dim) const noexcept {
                if (dim > 0)
                    return this->base_type::element_count(dim - 1);
                return element_count();
            }

            PBW_ANNOTATE__ size_type size() const noexcept {
                return bounds_;
            }

            PBW_ANNOTATE__ bool operator==(bounds_ranges const &rhs) const noexcept {
                return bounds_ == rhs.bounds_;
            }
        private:
            size_type bounds_ = 0;
        };

        template<std::ptrdiff_t CurRange, std::ptrdiff_t... RestRanges>
        struct bounds_ranges<CurRange, RestRanges...> : bounds_ranges<RestRanges...> {
            using base_type = bounds_ranges<RestRanges...>;
            using size_type = std::ptrdiff_t;
            static std::size_t const depth = base_type::depth + 1;
            static std::size_t const dynamic_num = base_type::dynamic_num;
            static size_type const current_range = CurRange;
            static size_type const total_size = dynamic_range == base_type::total_size ? dynamic_range : current_range * base_type::total_size;

            PBW_ANNOTATE__ bounds_ranges() = default;

            PBW_ANNOTATE__ bounds_ranges(std::ptrdiff_t const *array)
                : base_type(array) {}

            template<std::ptrdiff_t OtherRange, std::ptrdiff_t... RestOtherRanges>
            PBW_ANNOTATE__ bounds_ranges(bounds_ranges<OtherRange, RestOtherRanges...> const &other, bool first = true)
                : base_type(other, false)
            {
                (void)first;
            }

            template<typename T, std::size_t Dim = 0>
            PBW_ANNOTATE__ void serialize(T &array) const
            {
                array[Dim] = element_count();
                this->base_type::template serialize<T, Dim + 1>(array);
            }

            template<typename T, std::size_t Dim = 0>
            PBW_ANNOTATE__ size_type linearize(T const &array) const
            {
                return this->base_type::size() * array[Dim] + this->base_type::template linearize<T, Dim + 1>(array);
            }

            template<typename T, std::size_t Dim = 0>
            PBW_ANNOTATE__ size_type contains(T const &array) const
            {
                if (array[Dim] >= current_range)
                    return -1;
                size_type const last = this->base_type::template contains<T, Dim + 1>(array);
                if (last == -1)
                    return -1;
                return this->base_type::size() * array[Dim] + last;
            }

            PBW_ANNOTATE__ size_type size() const noexcept
            {
                return current_range * this->base_type::size();
            }

            PBW_ANNOTATE__ size_type element_count() const noexcept
            {
                return current_range;
            }

            PBW_ANNOTATE__ size_type element_count(std::size_t dim) const noexcept
            {
                if (dim > 0)
                    return this->base_type::element_count(dim - 1);
                return element_count();
            }
        };

        template<typename SourceType, typename TargetType>
        struct bounds_range_convertible : public std::conditional_t<
                (SourceType::total_size >= TargetType::total_size ||
                 TargetType::total_size == dynamic_range ||
                 SourceType::total_size == dynamic_range ||
                 TargetType::total_size == 0), std::false_type, std::true_type> {};

        template<typename TypeChain>
        struct type_list_indexer
        {
        private:
            TypeChain const& object_;
        public:
            PBW_ANNOTATE__ type_list_indexer(TypeChain const& obj)
                    : object_(obj)
            {}

            template<std::size_t N>
            PBW_ANNOTATE__ TypeChain const& get_object(std::true_type)
            {
                return object_;
            }

            template<std::size_t N, typename MyChain = TypeChain, typename BaseType = typename TypeChain::base_type>
            PBW_ANNOTATE__ auto get_object(std::false_type)
            -> decltype(type_list_indexer<BaseType>(static_cast<BaseType const&>(object_)).template get<N>())
            {
                return type_list_indexer<BaseType>(static_cast<BaseType const&>(object_)).template get<N>();
            }

            template<std::size_t N>
            PBW_ANNOTATE__ auto get()
            -> decltype(get_object<N - 1>(std::integral_constant<bool, N == 0>()))
            {
                return get_object<N - 1>(std::integral_constant<bool, N == 0>());
            }
        };

        template<typename TypeChain>
        PBW_ANNOTATE__ type_list_indexer<TypeChain> make_indexer(TypeChain const& obj)
        {
            return type_list_indexer<TypeChain>(obj);
        }
    }
}