// function<F> out of std:: if available, otherwise boost::
#pragma once

#include <geode/utility/config.h>

#if GEODE_HAS_CPP11_STD_HEADER(<functional>)
#include <functional>
#define GEODE_FUNCTION_NAMESPACE std
#else
#include <boost/function.hpp>
#define GEODE_FUNCTION_NAMESPACE boost
#endif

namespace geode {
using GEODE_FUNCTION_NAMESPACE::function;
}
