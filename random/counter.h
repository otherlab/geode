//#####################################################################
// Counter-based random numbers
//#####################################################################
#pragma once

#include <other/core/random/forward.h>
#include <other/core/math/uint128.h>
namespace other {

// Note that we put key first to match currying, unlike Salmon et al.
OTHER_CORE_EXPORT uint128_t threefry(uint128_t key, uint128_t ctr) OTHER_CONST;

}
