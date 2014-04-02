// Random access pseudorandom permutations

#include <geode/random/permute.h>
#include <geode/random/counter.h>
#include <geode/math/integer_log.h>
namespace geode {

// For details, see
//   John Black and Phillip Rogaway, Ciphers with Arbitrary Finite Domains.
//   Mihir Bellare, Thomas Ristenpart, Phillip Rogaway, and Till Stegers, Format-Preserving Encryption.
//   http://blog.notdot.net/2007/9/Damn-Cool-Algorithms-Part-2-Secure-permutations-with-block-ciphers
// Specifically, we use the FE2 construction with a = 2^k, b = 2^k or 2^(k+1), three rounds, and threefry
// as the round function.  Then we use cycle walking to turn this into a permutation on [0,n-1].

// An earlier version used TEA, but the quality of this was questionable and it didn't work for small n.

static inline uint64_t fe2_encrypt(const int bits, const uint128_t key, const uint64_t x) {
  assert(bits==64 || x<(uint64_t(1)<<bits));
  // Prepare for FE2
  const int a = bits>>1, b = bits-a; // logs of a,b in the paper
  const uint32_t ma = (uint32_t)(uint64_t(1)<<a)-1, // bit masks
                 mb = (uint32_t)(uint64_t(1)<<b)-1;
  uint32_t L = uint32_t(x>>b), R = x&mb;
  // Three rounds of FE2
  L = ma&(L+cast_uint128<uint32_t>(threefry(key,uint64_t(1)<<32|R))); // round 1: s = a
  R = mb&(R+cast_uint128<uint32_t>(threefry(key,uint64_t(2)<<32|L))); // round 2: s = b
  L = ma&(L+cast_uint128<uint32_t>(threefry(key,uint64_t(3)<<32|R))); // round 3: s = a
  return uint64_t(L)<<b|R;
}

// The inverse of fe2_encrypt
static inline uint64_t fe2_decrypt(const int bits, const uint128_t key, const uint64_t x) {
  assert(bits==64 || x<(uint64_t(1)<<bits));
  // Prepare for FE2
  const int a = bits>>1, b = bits-a; // logs of a,b in the paper
  const uint32_t ma = (uint32_t)(uint64_t(1)<<a)-1, // bit masks
                 mb = (uint32_t)(uint64_t(1)<<b)-1;
  uint32_t L = uint32_t(x>>b), R = x&mb;
  // Three rounds of FE2
  L = ma&(L-cast_uint128<uint32_t>(threefry(key,uint64_t(3)<<32|R))); // round 3: s = a
  R = mb&(R-cast_uint128<uint32_t>(threefry(key,uint64_t(2)<<32|L))); // round 2: s = b
  L = ma&(L-cast_uint128<uint32_t>(threefry(key,uint64_t(1)<<32|R))); // round 1: s = a
  return uint64_t(L)<<b|R;
}

// Find a power of two strictly larger than n.  By insisting on strictly larger, we avoid the weakness
// that Feistel permutations are always even (apparently).  This would be detectable for small n.
static inline int next_log(const uint64_t n) {
  return integer_log(n)+1;
}

// Since we use the next power of two for the FE2 block cipher, the cycle walking construction
// needs an average of at most 2 iterations.  This amounts to 6 calls to threefry, or a couple
// hundred cycles.

uint64_t random_permute(const uint64_t n, const uint128_t key, uint64_t x) {
  assert(x<n);
  const int bits = next_log(n);
  do { // Repeatedly encrypt until we're back in the right range
    x = fe2_encrypt(bits,key,x);
  } while (x>=n);
  return x;
}

uint64_t random_unpermute(const uint64_t n, const uint128_t key, uint64_t x) {
  assert(x<n);
  const int bits = next_log(n);
  do { // Repeatedly decrypt until we're back in the right range
    x = fe2_decrypt(bits,key,x);
  } while (x>=n);
  return x;
}

}
