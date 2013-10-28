// Random access pseudorandom permutations
#pragma once

#include <geode/math/uint128.h>
#include <geode/vector/Vector.h>
namespace geode {

// Apply a pseudorandom permutation to the range [0,n-1]
GEODE_CORE_EXPORT uint64_t random_permute(uint64_t n, uint128_t key, uint64_t x) GEODE_CONST;
GEODE_CORE_EXPORT uint64_t random_unpermute(uint64_t n, uint128_t key, uint64_t x) GEODE_CONST;

}
