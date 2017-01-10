// Pull type_traits out of std:: and define a few aliases and utilities
#pragma once

#include <geode/utility/config.h> // Must be included first
#include <geode/utility/mpl.h>

#include <cstdint>
#include <type_traits>
namespace geode {
template<class T> struct add_reference : public std::add_lvalue_reference<T> {};
template<bool c,class T=void> struct enable_if_c : public std::enable_if<c,T> {};
template<bool c,class T=void> struct disable_if_c : public std::enable_if<!c,T> {};
template<class C,class T=void> struct enable_if : public std::enable_if<C::value,T> {};
template<class C,class T=void> struct disable_if : public std::enable_if<!C::value,T> {};

using std::add_const;
using std::aligned_storage;
using std::alignment_of;
using std::common_type;
using std::declval;
using std::is_base_of;
using std::is_class;
using std::is_const;
using std::is_convertible;
using std::is_empty;
using std::is_enum;
using std::is_floating_point;
using std::is_fundamental;
using std::is_integral;
using std::is_pointer;
using std::is_reference;
using std::is_same;
using std::is_signed;
using std::is_unsigned;
using std::make_unsigned;
using std::remove_const;
using std::remove_pointer;
using std::remove_reference;
using std::result_of;

// Provide is_trivially_destructible as alias of has_trivial_destructor or manually implement it
#ifdef __GNUC__
// The following don't necessary exist on Mac, so implement them ourselves if possible
template<class T> struct is_trivially_destructible : public mpl::bool_<__has_trivial_destructor(T)> {};
#else
using std::is_trivially_destructible;
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

} // geode namespace
