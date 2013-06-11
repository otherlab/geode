// Multiprecision integer arithmetic for exact geometric predicates
#pragma once

#include <other/core/exact/config.h>
#include <other/core/math/integer_log.h>
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
    type n; \
    explicit Exact(const type n) : n(n) {} \
    template<int a> explicit Exact(const Exact<a> x) : n(x.n) { BOOST_STATIC_ASSERT(a<=d); } \
    void operator+=(const Exact x) { n += x.n; } \
  }; \
  OTHER_UNUSED static inline int sign(const Exact<d> x) { \
    return x.n<0?-1:x.n==0?0:1; \
  }
#if OTHER_EXACT_INT==32
OTHER_SMALL_EXACT(1,int32_t)
OTHER_SMALL_EXACT(2,int64_t)
OTHER_SMALL_EXACT(3,__int128_t)
OTHER_SMALL_EXACT(4,__int128_t)
#elif OTHER_EXACT_INT==64
OTHER_SMALL_EXACT(1,int64_t)
OTHER_SMALL_EXACT(2,__int128_t)
#endif
#undef OTHER_SMALL_EXACT

template<class I> struct IsSmall : public mpl::or_<boost::is_integral<I>,boost::is_same<I,__int128_t>> {};

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
  BOOST_STATIC_ASSERT(sizeof(Quantized)*d>128/8);
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

  template<class I> Exact(const I x, typename boost::enable_if<IsSmall<I>,Unusable>::type=Unusable()) {
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
#if OTHER_EXACT_INT==32
OTHER_EXACT_LOW_ADD(3,3)
OTHER_EXACT_LOW_ADD(4,4)
#endif
#define OTHER_EXACT_LOW_MUL(a,b) OTHER_EXACT_LOW_OP(*,a,b,a+b)
OTHER_EXACT_LOW_MUL(1,1)
#if OTHER_EXACT_INT==32
OTHER_EXACT_LOW_MUL(1,2)
OTHER_EXACT_LOW_MUL(2,1)
OTHER_EXACT_LOW_MUL(2,2)
#endif
#undef OTHER_EXACT_LOW_MUL
#undef OTHER_EXACT_LOW_ADD
#undef OTHER_EXACT_LOW_OP
#define OTHER_EXACT_LOW_NEG(a) \
  static inline Exact<a> operator-(const Exact<a> x) { \
    return Exact<a>(-x.n); \
  }
OTHER_EXACT_LOW_NEG(1)
OTHER_EXACT_LOW_NEG(2)
#if OTHER_EXACT_INT==32
OTHER_EXACT_LOW_NEG(3)
OTHER_EXACT_LOW_NEG(4)
#endif
#undef OTHER_EXACT_LOW_NEG

// Multiplications that turn two small Exact<a>'s into a large, or combine a small argument with a large

static inline void mpz_mul_helper(mpz_t r, mpz_srcptr x, mpz_srcptr y) {
  mpz_mul(r,x,y);
}

template<class I> static inline typename boost::enable_if<IsSmall<I>>::type mpz_mul_helper(mpz_t r, mpz_srcptr x, const I y) {
  if (sizeof(I)<=sizeof(long))
    mpz_mul_si(r,x,y);
  else
    mpz_mul(r,x,Exact<>(y).n);
}

template<class I> static inline typename boost::enable_if<IsSmall<I>>::type mpz_mul_helper(mpz_t r, const I x, mpz_srcptr y) {
  mpz_mul_helper(r,y,x);
}

template<class Ix,class Iy> static inline typename boost::enable_if<mpl::and_<IsSmall<Ix>,IsSmall<Iy>>>::type mpz_mul_helper(mpz_t r, const Ix x, const Iy y) {
  if (sizeof(x)>=sizeof(y))
    mpz_mul_helper(r,Exact<>(x).n,y);
  else
    mpz_mul_helper(r,Exact<>(y).n,x);
}

// Arithmetic for large Exact<d>

#define OTHER_EXACT_BIG_OP(op,f,ab) \
  template<int a,int b> static inline Exact<(ab)> operator op(const Exact<a>& x, const Exact<b>& y) { \
    Exact<(ab)> r; \
    f(r.n,x.n,y.n); \
    return other::move(r); \
  } \
  template<int a,int b> static inline typename boost::enable_if_c<Exact<a>::big,Exact<(ab)>>::type operator op(Exact<a>&& x, const Exact<b>& y) { \
    f(x.n,x.n,y.n); \
    return other::move(x); \
  } \
  template<int a,int b> static inline typename boost::enable_if_c<Exact<b>::big,Exact<(ab)>>::type operator op(const Exact<a>& x, Exact<b>&& y) { \
    f(y.n,x.n,y.n); \
    return other::move(y); \
  } \
  template<int a,int b> static inline typename boost::enable_if_c<Exact<a>::big,Exact<(ab)>>::type operator op(Exact<a>&& x, Exact<b>&& y) { \
    f(x.n,x.n,y.n); \
    return other::move(x); \
  }
OTHER_EXACT_BIG_OP(+,mpz_add,a>b?a:b)
OTHER_EXACT_BIG_OP(-,mpz_sub,a>b?a:b)
OTHER_EXACT_BIG_OP(*,mpz_mul_helper,a+b)
#undef OTHER_EXACT_BIG_OP

template<int a> static inline Exact<a> operator-(const Exact<a>& x) {
  Exact<a> r;
  mpz_neg(r.n,x.n);
  return other::move(r);
}

template<int a> static inline Exact<a> operator-(Exact<a>&& x) {
  mpz_neg(x.n,x.n);
  return other::move(x);
}

template<int a> static inline typename boost::enable_if_c<Exact<a>::big,Exact<2*a>>::type sqr(Exact<a>&& x) {
  mpz_mul(x.n,x.n,x.n);
  return other::move(x);
}

template<int a> static inline typename boost::enable_if_c<Exact<a>::big,Exact<3*a>>::type cube(Exact<a>&& x) {
  mpz_pow_ui(x.n,x.n,3);
  return other::move(x);
}

// Multiplication by small constants, assumed to not increase the precision required

template<int a> static inline typename boost::disable_if_c<Exact<a>::big,Exact<a>>::type small_mul(const int n, const Exact<a>& x) {
  return Exact<a>(n*x.n);
}

template<int a> static inline typename boost::enable_if_c<Exact<a>::big,Exact<a>>::type small_mul(const int n, const Exact<a>& x) {
  Exact<a> nx;
  if (power_of_two(uint32_t(abs(n)))) { // This routine will normally be inlined with constant n, so this check is cheap
    if (abs(n) > 1)
      mpz_mul_2exp(nx.n,x.n,integer_log_exact(uint32_t(abs(n))));
    if (n < 0)
      mpz_neg(nx.n);
  } else
    mpz_mul_si(nx.n,x.n,n);
  return other::move(nx);
}

template<int a> static inline typename boost::enable_if_c<Exact<a>::big,Exact<a>>::type small_mul(const int n, Exact<a>&& x) {
  if (power_of_two(uint32_t(abs(n)))) { // This routine will normally be inlined with constant n, so this check is cheap
    if (abs(n) > 1)
      mpz_mul_2exp(x.n,x.n,integer_log_exact(uint32_t(abs(n))));
    if (n < 0)
      mpz_neg(x.n,x.n);
  } else
    mpz_mul_si(x.n,x.n,n);
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
