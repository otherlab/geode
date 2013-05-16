// Multiprecision integer arithmetic for exact geometric predicates
#pragma once

#include <other/core/exact/config.h>
#include <other/core/math/uint128.h>
#include <other/core/utility/move.h>
#include <boost/detail/endian.hpp>
#include <boost/noncopyable.hpp>
#include <gmp.h>
namespace other {
namespace exact {

// Exact<d> holds 32*d bits for d<=4.  For d>4, Exact<d> thinly wraps a GMP mpz_t.
template<int degree=100> struct Exact;

// Small integers

#define OTHER_SMALL_EXACT(d,type_) \
  template<> struct Exact<d> { \
    static const bool big = false; \
    static const int degree = d; \
    typedef type_ type; \
    const type n; \
    explicit Exact(const type n) : n(n) {} \
  }; \
  OTHER_UNUSED static inline int sign(const Exact<d> x) { \
    return x.n<0?-1:x.n==0?0:1; \
  }
OTHER_SMALL_EXACT(1,int32_t)
OTHER_SMALL_EXACT(2,int64_t)
OTHER_SMALL_EXACT(3,__int128_t)
OTHER_SMALL_EXACT(4,__int128_t)
#undef OTHER_SMALL_EXACT

// Overloaded conversion from small integers to GMP

template<class I> static inline void init_set_steal(mpz_t x, const I n) {
  if (sizeof(I)<=sizeof(long))
    mpz_init_set_si(x,long(n));
  else {
    #if defined(BOOST_LITTLE_ENDIAN)
      static const int order = -1;
    #elif defined(BOOST_BIG_ENDIAN)
      static const int order =  1;
    #endif
    const I abs_n = n>=0?n:-n;
    mpz_init(x);
    mpz_import(x,sizeof(I)/8,order,8,0,0,&abs_n);
    if (n<0)
      mpz_neg(x,x);
  }
}

static inline void init_set_steal(mpz_t x, mpz_t y) {
  // We need to do x = y, y = detectable-uninitialized.  This way relies on the shape of mpz_t, which is bad but not too bad.
  *x = *y;
  y->_mp_d = 0;
}

template<int d> struct Exact : public boost::noncopyable {
  BOOST_STATIC_ASSERT(d>4);
  struct Unusable {};
  static const bool big = true;
  static const int degree = d;
  mpz_t n;

  Exact() {
    mpz_init(n);
  }

  // Make the copy constructors explicit to avoid accidental copies
  explicit Exact(const Exact& x) {
    mpz_init_set(n,x.n);
  }
  template<int k> explicit Exact(const Exact<k>& x, typename boost::enable_if_c<Exact<k>::big,Unusable>::type=Unusable()) {
    mpz_init_set(n,x.n);
  }
  template<int k> explicit Exact(const Exact<k>& x, typename boost::disable_if_c<Exact<k>::big,Unusable>::type=Unusable()) {
    init_set_steal(n,x.n);
  }

  template<int k> Exact(Exact<k>&& x) {
    init_set_steal(n,x.n);
  }

  template<class I> Exact(const I x) {
    init_set_steal(n,x);
  }

  ~Exact() {
    if (n->_mp_d)
      mpz_clear(n);
  }
};

template<int d> static inline int sign(const Exact<d>& x) {
  return mpz_sgn(x.n);
}

// Arithmetic for small Exact<d>

#define OTHER_EXACT_LOW_OP(op,a,b,ab) \
  static inline Exact<ab> operator op(const Exact<a> x, const Exact<b> y) { \
    return Exact<ab>(Exact<ab>::type(x.n) op y.n); \
  }
#define OTHER_EXACT_LOW_ADD(a,b) \
  OTHER_EXACT_LOW_OP(+,a,b,(a>b?a:b)) \
  OTHER_EXACT_LOW_OP(-,a,b,(a>b?a:b))
OTHER_EXACT_LOW_ADD(1,1)
OTHER_EXACT_LOW_ADD(2,2)
OTHER_EXACT_LOW_ADD(3,3)
OTHER_EXACT_LOW_ADD(4,4)
#define OTHER_EXACT_LOW_MUL(a,b) OTHER_EXACT_LOW_OP(*,a,b,a+b)
OTHER_EXACT_LOW_MUL(1,1)
OTHER_EXACT_LOW_MUL(1,2)
OTHER_EXACT_LOW_MUL(2,1)
OTHER_EXACT_LOW_MUL(2,2)
#undef OTHER_EXACT_LOW_MUL
#undef OTHER_EXACT_LOW_ADD
#undef OTHER_EXACT_LOW_OP

// Arithmetic for large Exact<d>

#define OTHER_EXACT_BIG_OP(op,f,ab) \
  template<int a,int b> static inline Exact<(ab)> operator op(const Exact<a>& x, const Exact<b>& y) { \
    Exact<(ab)> r; \
    f(r.n,x.n,y.n); \
    return other::move(r); \
  } \
  template<int a,int b> static inline Exact<(ab)> operator op(Exact<a>&& x, const Exact<b>& y) { \
    f(x.n,x.n,y.n); \
    return other::move(x); \
  } \
  template<int a,int b> static inline Exact<(ab)> operator op(const Exact<a>& x, Exact<b>&& y) { \
    f(y.n,y.n,x.n); \
    return other::move(y); \
  } \
  template<int a,int b> static inline Exact<(ab)> operator op(Exact<a>&& x, Exact<b>&& y) { \
    f(x.n,x.n,y.n); \
    return other::move(x); \
  }
OTHER_EXACT_BIG_OP(+,mpz_add,a>b?a:b)
OTHER_EXACT_BIG_OP(-,mpz_sub,a>b?a:b)
OTHER_EXACT_BIG_OP(*,mpz_mul,a+b)
#undef OTHER_EXACT_BIG_OP

template<int a> static inline Exact<2*a> sqr(Exact<a>&& x) {
  mpz_mul(x.n,x.n,x.n);
  return other::move(x);
}

template<int a> static inline Exact<3*a> cube(Exact<a>&& x) {
  mpz_pow_ui(x.n,x.n,3);
  return other::move(x);
}

// Stream output

}

OTHER_CORE_EXPORT ostream& operator<<(ostream& output, mpz_t x);
OTHER_CORE_EXPORT ostream& operator<<(ostream& output, mpq_t x);
static inline ostream& operator<<(ostream& output, __mpz_struct& x) { return output << &x; }
static inline ostream& operator<<(ostream& output, __mpq_struct& x) { return output << &x; }

namespace exact {

template<int d> static inline ostream& operator<<(ostream& output, const Exact<d>& x) {
  using other::operator<<;
  return output << x.n;
}

}
}
