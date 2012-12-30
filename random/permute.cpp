// Random access pseudorandom permutations

#include <other/core/random/permute.h>
#include <other/core/math/integer_log.h>
#include <other/core/python/module.h>
namespace other {

// For details, see
//   http://en.wikipedia.org/wiki/Tiny_Encryption_Algorithm
//   http://blog.notdot.net/2007/9/Damn-Cool-Algorithms-Part-2-Secure-permutations-with-block-ciphers
//   Black and Rogaway, Ciphers with Arbitrary Finite Domains.
// Based on code released into the public domain by David Wheeler and Roger Needham.

// Low bit width TEA block cypher, acting as a permutation on [0,(1<<2*half_bits)-1].  Requires half_bits in [6,32].
// I am not sure whether all special keys give reasonable permutations, so choose a random one to be safe.
// Warning: TEA has a variety of cryptographic weaknesses.

static const int rounds = 32;
static const uint32_t delta = 0x9e3779b9;

static inline uint64_t tea_encrypt(const int half_bits, const uint128_t key, const uint64_t x) {
  assert(6<=half_bits && half_bits<=32);
  const uint32_t k0 = cast_uint128<uint32_t>(key),
                 k1 = cast_uint128<uint32_t>(key>>32),
                 k2 = cast_uint128<uint32_t>(key>>64),
                 k3 = cast_uint128<uint32_t>(key>>96),
                 mask((uint64_t(1)<<half_bits)-1);
  assert(half_bits==32 || !(x>>2*half_bits));
  uint32_t x0(x&mask),
           x1(x>>half_bits),
           sum = 0;
  for (int i=0;i<rounds;i++) {
    sum += delta;
    x0 = mask&(x0+(((x1<<4)+k0)^(x1+sum)^((x1>>5)+k1)));
    x1 = mask&(x1+(((x0<<4)+k2)^(x0+sum)^((x0>>5)+k3)));
  }
  return x0|uint64_t(x1)<<half_bits;
}

// The inverse of tea_encrypt
static inline uint64_t tea_decrypt(const int half_bits, const uint128_t key, const uint64_t x) {
  assert(6<=half_bits && half_bits<=32);
  const uint32_t k0 = cast_uint128<uint32_t>(key),
                 k1 = cast_uint128<uint32_t>(key>>32),
                 k2 = cast_uint128<uint32_t>(key>>64),
                 k3 = cast_uint128<uint32_t>(key>>96),
                 mask((uint64_t(1)<<half_bits)-1);
  assert(half_bits==32 || !(x>>2*half_bits));
  uint32_t x0(x&mask),
           x1(x>>half_bits),
           sum = rounds*delta;
  for (int i=0;i<rounds;i++) {
    x1 = mask&(x1-(((x0<<4)+k2)^(x0+sum)^((x0>>5)+k3)));
    x0 = mask&(x0-(((x1<<4)+k0)^(x1+sum)^((x1>>5)+k1)));
    sum -= delta;
  }
  return x0|uint64_t(x1)<<half_bits;
}

// Since we use the next power of four for the TEA block cypher, the cycle walking construction
// needs an average of at most 4 iterations.  If n is chosen logarithmically at random, the
// average iteration count is 3/2 log 2 = 2.164...

static int ceil_half_log(const uint64_t n) {
  if (n > (uint64_t(1)<<62))
    return 32;
  const int b = (integer_log(2*n-1)+1)>>1;
  OTHER_DEBUG_ONLY(const auto hi = uint64_t(1)<<2*b;)
  assert(hi/4<n && n<=hi);
  return b;
}

uint64_t random_permute(const uint64_t n, const uint128_t key, uint64_t x) {
  OTHER_ASSERT(n>1024 && x<n);
  const int half_bits = ceil_half_log(n);
  do { // Repeatedly encrypt until we're back in the right range
    x = tea_encrypt(half_bits,key,x);
  } while (x>=n);
  return x;
}

uint64_t random_unpermute(const uint64_t n, const uint128_t key, uint64_t x) {
  OTHER_ASSERT(n>1024 && x<n);
  const int half_bits = ceil_half_log(n);
  do { // Repeatedly decrypt until we're back in the right range
    x = tea_decrypt(half_bits,key,x);
  } while (x>=n);
  return x;
}

}
using namespace other;

void wrap_permute() {
  OTHER_FUNCTION(tea_encrypt)
  OTHER_FUNCTION(tea_decrypt)
  OTHER_FUNCTION(random_permute)
  OTHER_FUNCTION(random_unpermute)
}
