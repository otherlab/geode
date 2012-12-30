// Random access pseudorandom permutations
#pragma once

#include <other/core/math/uint128.h>
#include <other/core/vector/Vector.h>
namespace other {

// Apply a pseudorandom permutation to the range [0,n-1].
// I am not sure whether all special keys give reasonable permutations, so choose a random one to be safe.
OTHER_CORE_EXPORT uint64_t random_permute(uint64_t n, uint128_t key, uint64_t x) OTHER_CONST;
OTHER_CORE_EXPORT uint64_t random_unpermute(uint64_t n, uint128_t key, uint64_t x) OTHER_CONST;

}
