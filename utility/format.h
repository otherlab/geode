//#####################################################################
// Function format
//#####################################################################
//
// Similar to boost::format, with the following differences:
//
// 1. Not as powerful (no format("%g")%vector) or as safe.
// 2. More concise.
// 3. Doesn't add 100k to every object file.
//
// The main advantage over raw varargs is that we can pass string, and
// we can add more safety features later if we feel like it.
//
//#####################################################################
#pragma once

#include <other/core/utility/config.h>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/or.hpp>
#include <boost/type_traits/is_fundamental.hpp>
#include <boost/type_traits/is_enum.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/utility/enable_if.hpp>
#include <string>
namespace other {

namespace mpl = boost::mpl;
using std::string;

// Unfortunately, since format_helper is called indirectly through format, we can't use gcc's format attribute.
string format_helper(const char* format,...) OTHER_EXPORT;

template<class T> static inline typename mpl::if_<boost::is_enum<T>,int,T>::type format_sanitize(const T d) {
  // Ensure that passing as a vararg is safe
  BOOST_MPL_ASSERT((mpl::or_<boost::is_fundamental<T>,boost::is_enum<T>,boost::is_pointer<T>>));
  return d;
}

static inline const char* format_sanitize(char* s) {
  return s;
}

static inline const char* format_sanitize(const char* s) {
  return s;
}

static inline const char* format_sanitize(const string& s) {
  return s.c_str();
}

template<class... Args> static inline string format(const char* format, const Args&... args) {
  return format_helper(format,format_sanitize(args)...);
}

}
