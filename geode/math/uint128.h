//#####################################################################
// Class uint128_t
//#####################################################################
#pragma once

#include <geode/utility/type_traits.h>
#include <stdint.h>
#include <string>
#include <vector>
namespace geode {

using std::string;
using std::ostream;
using std::vector;

#if defined(__GNUC__) && defined(__LP64__)

// Use the native integer type if possible
typedef  __int128_t  int128_t;
typedef __uint128_t uint128_t;

template<class I> static inline I cast_uint128(const uint128_t& n) {
  return I(n);
}

#else

class uint128_t {
#if GEODE_ENDIAN == GEODE_LITTLE_ENDIAN
  uint64_t lo,hi;
#elif GEODE_ENDIAN == GEODE_BIG_ENDIAN
  uint64_t hi,lo;
#endif
public:

  uint128_t()
    :lo(),hi() {}

  uint128_t(uint32_t n)
    :lo(n),hi() {}

  uint128_t(uint64_t n)
    :lo(n),hi() {}

  uint128_t(int32_t n)
    :lo(n),hi(n<0?-1:0) {}

  uint128_t(int64_t n)
    :lo(n),hi(n<0?-1:0) {}

private:
  // Private since this constructor doesn't exist if we're using the native __uint128_t version
  uint128_t(uint64_t hi,uint64_t lo)
    :lo(lo),hi(hi) {}
public:

  bool operator==(uint128_t x) const {
    return lo==x.lo && hi==x.hi;
  }

  bool operator!=(uint128_t x) const {
    return lo!=x.lo || hi!=x.hi;
  }

  uint128_t operator+(uint128_t x) const {
    uint64_t l = lo+x.lo;
    return uint128_t(hi+x.hi+(l<lo),l);
  }

  uint128_t operator-(uint128_t x) const {
    uint64_t l = lo-x.lo;
    return uint128_t(hi-x.hi-(l>lo),l);
  }

  uint128_t operator*(uint128_t x) const {
    // Use 64-bit multiplies, since we're assuming a native uint128_t isn't available
    uint64_t mask = (uint64_t(1)<<32)-1, ll = lo&mask, xll = x.lo&mask, lh = lo>>32, xlh = x.lo>>32, m0 = xlh*ll, m1 = xll*lh;
    return uint128_t(lo*x.hi+hi*x.lo+xlh*lh,ll*xll)+uint128_t(m0>>32,m0<<32)+uint128_t(m1>>32,m1<<32);
  }

  uint128_t operator<<(unsigned b) const {
    return b==0?*this:b<64?uint128_t(hi<<b|lo>>(64-b),lo<<b):uint128_t(lo<<(b-64),0);
  }

  uint128_t operator>>(unsigned b) const {
    return b==0?*this:b<64?uint128_t(hi>>b,lo>>b|hi<<(64-b)):uint128_t(0,hi>>(b-64));
  }

  uint128_t& operator<<=(unsigned b) {
    *this = *this<<b;
    return *this;
  }

  uint128_t& operator>>=(unsigned b) {
    *this = *this>>b;
    return *this;
  }

  uint64_t operator&(uint64_t x) const {
    return lo&x;
  }

  uint128_t operator&(uint128_t x) const {
    return uint128_t(hi&x.hi,lo&x.lo);
  }

  uint128_t operator|(uint128_t x) const {
    return uint128_t(hi|x.hi,lo|x.lo);
  }

  uint128_t operator~() const {
    return uint128_t(~hi,~lo);
  }

  uint128_t& operator++() { // prefix
    uint64_t l = lo;
    lo++;
    hi += lo<l;
    return *this;
  }

  uint128_t operator++(int) { // postfix
    uint128_t save(*this);
    ++*this;
    return save;
  }

  template<class I> friend inline I cast_uint128(const uint128_t& n);
};

template<class I> inline I cast_uint128(const uint128_t& n) {
  static_assert(is_integral<I>::value && sizeof(I)<=8,"");
  return I(n.lo);
}

#endif

template<> struct uint_t<128> { typedef uint128_t exact; };

GEODE_CORE_EXPORT string str(uint128_t n);
GEODE_CORE_EXPORT ostream& operator<<(ostream& output, uint128_t n);

#if defined(__GNUC__) && defined(__LP64__)
GEODE_CORE_EXPORT string str(int128_t n);
GEODE_CORE_EXPORT ostream& operator<<(ostream& output, int128_t n);
#endif

// For testing purposes
GEODE_CORE_EXPORT vector<uint128_t> uint128_test(uint128_t x, uint128_t y);

}
