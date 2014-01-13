// Pull type_traits out of std:: if available, otherwise boost::
#pragma once

#include <geode/utility/config.h> // Must be included first

// If we're on clang, check for the right header directly.  If we're not,
// any sufficient recently version of gcc should always have the right header.
#if defined(__clang__) ? __has_include(<type_traits>) : defined(__GNU__)

#include <type_traits>
#define GEODE_TYPE_TRAITS std
namespace geode {
template<class T> struct add_reference : public std::add_lvalue_reference<T> {};
template<bool c,class T=void> struct enable_if_c : public std::enable_if<c,T> {};
template<bool c,class T=void> struct disable_if_c : public std::enable_if<!c,T> {};
template<class C,class T=void> struct enable_if : public std::enable_if<C::value,T> {};
template<class C,class T=void> struct disable_if : public std::enable_if<!C::value,T> {};
}

#else

// No <type_traits> header, so pull everything out of boost.  We need at least boost-1.47.
#if BOOST_VERSION < 104700
#error "Since we lack the C++11 header <type_traits>, we need at least boost 1.47 for correct type traits."
#endif

#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/add_reference.hpp>
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
#define GEODE_TYPE_TRAITS boost
namespace geode {
using GEODE_TYPE_TRAITS::add_reference;
using GEODE_TYPE_TRAITS::disable_if;
using GEODE_TYPE_TRAITS::disable_if_c;
using GEODE_TYPE_TRAITS::enable_if;
using GEODE_TYPE_TRAITS::enable_if_c;
}

#endif

namespace geode {

using GEODE_TYPE_TRAITS::add_const;
using GEODE_TYPE_TRAITS::alignment_of;
using GEODE_TYPE_TRAITS::common_type;
using GEODE_TYPE_TRAITS::declval;
using GEODE_TYPE_TRAITS::has_trivial_destructor;
using GEODE_TYPE_TRAITS::is_base_of;
using GEODE_TYPE_TRAITS::is_class;
using GEODE_TYPE_TRAITS::is_const;
using GEODE_TYPE_TRAITS::is_convertible;
using GEODE_TYPE_TRAITS::is_empty;
using GEODE_TYPE_TRAITS::is_enum;
using GEODE_TYPE_TRAITS::is_floating_point;
using GEODE_TYPE_TRAITS::is_fundamental;
using GEODE_TYPE_TRAITS::is_integral;
using GEODE_TYPE_TRAITS::is_pointer;
using GEODE_TYPE_TRAITS::is_reference;
using GEODE_TYPE_TRAITS::is_same;
using GEODE_TYPE_TRAITS::remove_const;
using GEODE_TYPE_TRAITS::remove_pointer;
using GEODE_TYPE_TRAITS::remove_reference;
using GEODE_TYPE_TRAITS::result_of;

// Add a few more useful ones
template<class T> struct remove_const_reference : public remove_const<typename remove_reference<T>::type> {};

}
