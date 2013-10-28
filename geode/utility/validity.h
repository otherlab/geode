// Check whether expressions are valid at compile time
#pragma once

// For example, we can check whether a type can be multiplied by two as follows:
//
//   GEODE_VALIDITY_CHECKER(has_times_two,T,2*(*(T*)0));
//   BOOST_STATIC_ASSERT(has_times_two<int>::value);
//   BOOST_STATIC_ASSERT(!has_times_two<void>::value);
//
// See http://stackoverflow.com/questions/2127693/sfinae-sizeof-detect-if-expression-compiles

#include <geode/utility/config.h>
#include <geode/utility/forward.h>
#include <boost/mpl/bool.hpp>

#define GEODE_VALIDITY_CHECKER(name,T,expression) \
  template<class T> static char name##_helper(typename geode::First<int*,decltype(expression)>::type) { return 0; } /* success */ \
  template<class T> static long name##_helper(...) { return 0; } /* failure */ \
  template<class T> struct name : public boost::mpl::bool_<sizeof(name##_helper<T>(0))==1> {};
