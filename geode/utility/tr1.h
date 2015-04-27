// Include unordered_set and unordered_map in the appropriate manner
#pragma once

#include <geode/utility/config.h>

// If we're on clang, check for the right header directly.  If we're not,
// any sufficient recently version of gcc should always have the right header.
#if defined(__clang__) ? GEODE_HAS_INCLUDE(<unordered_set>) : defined(__GNUC__)
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
