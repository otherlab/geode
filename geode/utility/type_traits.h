// Pull type_traits out of std:: if available, otherwise boost::
#pragma once

#include <geode/utility/config.h> // Must be included first
#include <geode/utility/mpl.h>
#include <cstdint>

// If we're on clang, check for the right header directly.  If we're not,
// any sufficient recently version of gcc should always have the right header.
#if defined(__clang__) ? GEODE_HAS_INCLUDE(<type_traits>) : defined(__GNU__)

#include <type_traits>
#define GEODE_TYPE_TRAITS_NAMESPACE std
namespace geode {
template<class T> struct add_reference : public std::add_lvalue_reference<T> {};
template<bool c,class T=void> struct enable_if_c : public std::enable_if<c,T> {};
template<bool c,class T=void> struct disable_if_c : public std::enable_if<!c,T> {};
template<class C,class T=void> struct enable_if : public std::enable_if<C::value,T> {};
template<class C,class T=void> struct disable_if : public std::enable_if<!C::value,T> {};
}

#else

// No <type_traits> header, so pull everything out of boost.  We need at least boost-1.47.
#include <boost/version.hpp>
#if BOOST_VERSION < 104700
#error "Since we lack the C++11 header <type_traits>, we need at least boost 1.47 for correct type traits."
#endif

#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/add_reference.hpp>
#include <boost/type_traits/aligned_storage.hpp>
#include <boost/type_traits/alignment_of.hpp>
#include <boost/type_traits/common_type.hpp>
#include <boost/type_traits/has_trivial_destructor.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/is_class.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/is_empty.hpp>
#include <boost/type_traits/is_enum.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/type_traits/is_fundamental.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/is_reference.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/utility/declval.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/utility/result_of.hpp>
#define GEODE_TYPE_TRAITS_NAMESPACE boost
namespace geode {
using GEODE_TYPE_TRAITS_NAMESPACE::add_reference;
using GEODE_TYPE_TRAITS_NAMESPACE::disable_if;
using GEODE_TYPE_TRAITS_NAMESPACE::disable_if_c;
using GEODE_TYPE_TRAITS_NAMESPACE::enable_if;
using GEODE_TYPE_TRAITS_NAMESPACE::enable_if_c;
}

#endif

namespace geode {

using GEODE_TYPE_TRAITS_NAMESPACE::add_const;
using GEODE_TYPE_TRAITS_NAMESPACE::aligned_storage;
using GEODE_TYPE_TRAITS_NAMESPACE::alignment_of;
using GEODE_TYPE_TRAITS_NAMESPACE::common_type;
using GEODE_TYPE_TRAITS_NAMESPACE::declval;
using GEODE_TYPE_TRAITS_NAMESPACE::is_base_of;
using GEODE_TYPE_TRAITS_NAMESPACE::is_class;
using GEODE_TYPE_TRAITS_NAMESPACE::is_const;
using GEODE_TYPE_TRAITS_NAMESPACE::is_convertible;
using GEODE_TYPE_TRAITS_NAMESPACE::is_empty;
using GEODE_TYPE_TRAITS_NAMESPACE::is_enum;
using GEODE_TYPE_TRAITS_NAMESPACE::is_floating_point;
using GEODE_TYPE_TRAITS_NAMESPACE::is_fundamental;
using GEODE_TYPE_TRAITS_NAMESPACE::is_integral;
using GEODE_TYPE_TRAITS_NAMESPACE::is_pointer;
using GEODE_TYPE_TRAITS_NAMESPACE::is_reference;
using GEODE_TYPE_TRAITS_NAMESPACE::is_same;
using GEODE_TYPE_TRAITS_NAMESPACE::remove_const;
using GEODE_TYPE_TRAITS_NAMESPACE::remove_pointer;
using GEODE_TYPE_TRAITS_NAMESPACE::remove_reference;
using GEODE_TYPE_TRAITS_NAMESPACE::result_of;

// The following don't necessary exist on Mac, so implement them ourselves if possible
#ifdef __GNUC__
template<class T> struct has_trivial_destructor : public mpl::bool_<__has_trivial_destructor(T)> {};
#else
using GEODE_TYPE_TRAITS_NAMESPACE::has_trivial_destructor;
#endif

// Add a few more useful ones
template<class T> struct remove_const_reference : public remove_const<typename remove_reference<T>::type> {};

// Detect smart (or normal) pointers
template<class T> struct is_smart_pointer : public is_pointer<T> {};

// Unsigned ints with the right number of bits
template<int bits> struct uint_t;
#define UINT(bits) template<> struct uint_t<bits> { typedef uint##bits##_t exact; };
UINT(8)
UINT(16)
UINT(32)
UINT(64)
#undef UINT

}
