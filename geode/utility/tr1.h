// Include unordered_set and unordered_map in the appropriate manner
#pragma once

#include <geode/utility/config.h>

#if GEODE_HAS_CPP11_STD_HEADER(<unordered_set>) && GEODE_HAS_CPP11_STD_HEADER(<unordered_map>)
#include <unordered_set>
#include <unordered_map>
#define GEODE_UNORDERED_NAMESPACE std
#else
#include <boost/tr1/unordered_set.hpp>
#include <boost/tr1/unordered_map.hpp>
#define GEODE_UNORDERED_NAMESPACE boost
#endif

namespace geode {

using GEODE_UNORDERED_NAMESPACE::unordered_set;
using GEODE_UNORDERED_NAMESPACE::unordered_map;
using GEODE_UNORDERED_NAMESPACE::unordered_multiset;
using GEODE_UNORDERED_NAMESPACE::unordered_multimap;

}
