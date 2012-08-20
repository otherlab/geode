//#####################################################################
// Class uint128_t
//#####################################################################
#pragma once

#include <other/core/python/forward.h>
#include <stdint.h>
namespace other {

#if defined(__GNUC__) && defined(__x86_64__)

// Use the native integer type if possible
typedef __uint128_t uint128_t;

#else

class uint128_t {
  uint64_t lo,hi;
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
    uint64_t mask = (1L<<32)-1, ll = lo&mask, xll = x.lo&mask, lh = lo>>32, xlh = x.lo>>32, m0 = xlh*ll, m1 = xll*lh;
    return uint128_t(lo*x.hi+hi*x.lo+xlh*lh,ll*xll)+uint128_t(m0>>32,m0<<32)+uint128_t(m1>>32,m1<<32);
  }

  uint128_t operator<<(unsigned b) const {
    return b==64?uint128_t(lo,0):uint128_t((hi<<b)+(lo>>(64-b)),(lo<<b));
  }

  uint128_t operator>>(unsigned b) const {
    return b==64?uint128_t(0,hi):uint128_t(hi>>b,(lo>>b)+(hi<<(64-b)));
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
};

#endif

PyObject* to_python(uint128_t n) OTHER_EXPORT;
template<> struct FromPython<uint128_t>{OTHER_EXPORT static uint128_t convert(PyObject* object);};

}
