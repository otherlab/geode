// function<F> out of std:: if available, otherwise boost::
#pragma once

#include <geode/utility/config.h>

// If we're on clang, check for the right header directly.  If we're not,
// any sufficient recently version of gcc should always have the right header.
#if defined(__clang__) ? GEODE_HAS_INCLUDE(<functional>) : defined(__GNUC__)
#include <functional>
#define GEODE_FUNCTION_NAMESPACE std
#else
#include <boost/function.hpp>
#define GEODE_FUNCTION_NAMESPACE boost
#endif

namespace geode {
using GEODE_FUNCTION_NAMESPACE::function;
}
