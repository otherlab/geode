// Multiprecision integer arithmetic for exact geometric predicates
#pragma once

#include <other/core/exact/config.h>
#include <other/core/math/uint128.h>
#include <boost/detail/endian.hpp>
#include <gmpxx.h>
namespace other {
namespace exact {

// Exact<d> holds 32*d bits for d<=4.  For d>4, Exact<d> thinly wraps a GMP mpz_class.
template<int degree> struct Exact;

#define OTHER_SMALL_EXACT(d,type_) \
  template<> struct Exact<d> { \
    static const int degree = d; \
    typedef type_ type; \
    const type n; \
    explicit Exact(const type n) : n(n) {} \
    friend int sign(const Exact x) { return x.n<0?-1:x.n==0?0:1; } \
  };
OTHER_SMALL_EXACT(1,int32_t)
OTHER_SMALL_EXACT(2,int64_t)
OTHER_SMALL_EXACT(3,__int128_t)
OTHER_SMALL_EXACT(4,__int128_t)
#undef OTHER_SMALL_EXACT

template<int d> struct Exact {
  BOOST_STATIC_ASSERT(d>4);
  static const int degree = d;
  // TODO: Probably switch to GMP's C api to avoid copying overhead
  typedef mpz_class type;
  mpz_class n;

  template<class T> explicit Exact(const T& n)
    : n(n) {}

  // order parameter for mpz_import
#if defined(BOOST_LITTLE_ENDIAN)
  static const int order = -1;
#elif defined(BOOST_BIG_ENDIAN)
  static const int order =  1;
#endif

  // Need a special version for int64_t in case we're on a 32-bit machine
  explicit Exact(int64_t n_) {
    if (sizeof(int64_t)==sizeof(long))
      n = long(n_);
    else {
      const uint64_t an = abs(n_);
      mpz_import(n.get_mpz_t(),2,order,4,0,0,&an);
      if (n_<0)
        mpz_neg(n.get_mpz_t(),n.get_mpz_t());
    }
  }

  // We always need a special version for int64_t
  explicit Exact(__int128_t n_) {
    const uint128_t an = n_>=0?n_:-n_;
    mpz_import(n.get_mpz_t(),2,order,8,0,0,&an);
    if (n_<0)
      mpz_neg(n.get_mpz_t(),n.get_mpz_t());
  }

  friend int sign(const Exact& x) {
    return sgn(x.n);
  }
};

template<int a,int b> static inline Exact<(a>b?a:b)> operator+(const Exact<a>& x, const Exact<b>& y) {
  typedef Exact<(a>b?a:b)> Result;
  return Result(typename Result::type(x.n)+typename Result::type(y.n));
}

template<int a,int b> static inline Exact<(a>b?a:b)> operator-(const Exact<a>& x, const Exact<b>& y) {
  typedef Exact<(a>b?a:b)> Result;
  return Result(typename Result::type(x.n)-typename Result::type(y.n));
}

template<int a,int b> static inline Exact<a+b> operator*(const Exact<a>& x, const Exact<b>& y) {
  typedef Exact<a+b> Result;
  return Result(typename Result::type(x.n)*typename Result::type(y.n));
}

template<int a> static inline Exact<2*a> sqr(const Exact<a>& x) {
  typedef Exact<2*a> Result;
  return Result(typename Result::type(x.n)*x.n);
}

template<int a> static inline Exact<3*a> cube(const Exact<a>& x) {
  typedef Exact<3*a> Result;
  return Result(typename Result::type(x.n)*x.n*x.n);
}

template<int a> static inline ostream& operator<<(ostream& output, const Exact<a>& x) {
  using other::operator<<;
  return output << x.n;
}

}
}
