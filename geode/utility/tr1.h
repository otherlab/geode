// Include unordered_set and unordered_map in the appropriate manner
#pragma once

#include <geode/utility/config.h>

#ifdef _WIN32
#include <boost/tr1/unordered_set.hpp>
#include <boost/tr1/unordered_map.hpp>
#else
#include <tr1/unordered_set>
#include <tr1/unordered_map>
#endif

namespace geode {

using std::tr1::unordered_set;
using std::tr1::unordered_map;

}
