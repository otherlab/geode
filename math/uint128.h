//#####################################################################
// Class uint128_t
//#####################################################################
#pragma once

#include <other/core/python/forward.h>
#include <boost/type_traits/is_integral.hpp>
#include <boost/static_assert.hpp>
#include <stdint.h>
#include <string>
namespace other {

using std::string;
using std::ostream;

#if defined(__GNUC__) && defined(__x86_64__)

// Use the native integer type if possible
typedef __uint128_t uint128_t;

template<class I> static inline I cast_uint128(const uint128_t& n) {
  return I(n);
}

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
  BOOST_STATIC_ASSERT((boost::is_integral<I>::value && sizeof(I)<=8));
  return I(n.lo);
}

#endif

OTHER_CORE_EXPORT string str(uint128_t n);
OTHER_CORE_EXPORT ostream& operator<<(ostream& output, uint128_t n);
OTHER_CORE_EXPORT PyObject* to_python(uint128_t n);
template<> struct FromPython<uint128_t>{OTHER_CORE_EXPORT static uint128_t convert(PyObject* object);};

}
