// Check whether expressions are valid at compile time
#pragma once

// For example, we can check whether a type can be multiplied by two as follows:
//
//   OTHER_VALIDITY_CHECKER(has_times_two,T,2*(*(T*)0));
//   BOOST_STATIC_ASSERT(has_times_two<int>::value);
//   BOOST_STATIC_ASSERT(!has_times_two<void>::value);
//
// See http://stackoverflow.com/questions/2127693/sfinae-sizeof-detect-if-expression-compiles

#include <other/core/utility/config.h>
#include <other/core/utility/forward.h>
#include <boost/mpl/bool.hpp>

#define OTHER_VALIDITY_CHECKER(name,T,expression) \
  template<class T> static char name##_helper(typename other::First<int*,decltype(expression)>::type) { return 0; } /* success */ \
  template<class T> static long name##_helper(...) { return 0; } /* failure */ \
  template<class T> struct name : public boost::mpl::bool_<sizeof(name##_helper<T>(0))==1> {};
