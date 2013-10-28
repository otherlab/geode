//#####################################################################
// Counter-based random numbers
//#####################################################################
#pragma once

#include <geode/random/forward.h>
#include <geode/math/uint128.h>
namespace geode {

// Note that we put key first to match currying, unlike Salmon et al.
GEODE_CORE_EXPORT uint128_t threefry(uint128_t key, uint128_t ctr) GEODE_CONST;

}
