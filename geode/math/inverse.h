//#####################################################################
// Function inverse
//#####################################################################
#pragma once

#include <geode/utility/type_traits.h>
#include <cmath>
#include <cfloat>
#include <cassert>
namespace geode {

using ::std::abs;

template<class T> static inline auto inverse(const T& x)
  -> typename disable_if<is_fundamental<T>,decltype(x.inverse())>::type {
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

template<class I> static inline typename enable_if<is_integral<I>,IntInverse<I>>::type inverse(const I x) {
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
