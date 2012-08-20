//#####################################################################
// Function inverse
//#####################################################################
#pragma once

#include <cmath>
#include <cfloat>
#include <cassert>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_fundamental.hpp>
#include <boost/type_traits/is_integral.hpp>
namespace other {

using ::std::abs;

template<class TMatrix> static inline typename boost::disable_if<boost::is_fundamental<TMatrix>,TMatrix>::type inverse(const TMatrix& x) {
  return x.inverse();
}

static inline float inverse(const float x) {
  assert(abs(x)>=FLT_MIN);
  return 1/x;
}

static inline double inverse(const double x) {
  assert(abs(x)>=DBL_MIN);
  return 1/x;
}

template<class I> struct IntInverse {
  I a;
  IntInverse(I a) : a(a) {}
};

template<class I> static inline typename boost::enable_if<boost::is_integral<I>,IntInverse<I>>::type inverse(const I x) {
  assert(x!=0);
  return IntInverse<I>(x);
}

template<class I> static inline I operator*(const I x, const IntInverse<I> y) {
  return x/y.a;
}

template<class I> inline I& operator*=(I& x, const IntInverse<I> y) {
  return x/=y.a;
}

}
