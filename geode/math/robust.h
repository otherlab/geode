//#####################################################################
// Header robust
//#####################################################################
#pragma once

#include <cmath>
#include <cfloat>
#include <geode/math/small_sort.h>
#include <geode/math/sqr.h>
namespace geode {

using ::std::abs;
using ::std::atan2;
using ::std::sin;

template<class T> inline T robust_multiply(const T a, const T b) {
  T abs_a=abs(a),abs_b=abs(b);
  if (abs_b<1 || abs_a<FLT_MAX/abs_b) return a*b;
  else return (a>0)==(b>=0)?FLT_MAX:-FLT_MAX;
}

template<class T> inline T robust_divide(const T a, const T b) {
  T abs_a=abs(a),abs_b=abs(b);
  if (abs_b==FLT_MAX) return T();
  if (abs_b>1 || abs_a<FLT_MAX*abs_b) return a/b;
  else return (a>0)==(b>=0)?FLT_MAX:-FLT_MAX;
}

template<class T> inline T robust_inverse(const T a) {
  T abs_a=abs(a);
  if(abs_a==FLT_MAX) return T();
  if(abs_a>1 || FLT_MAX*abs_a>1) return 1/a;
  else return (a>=0)?FLT_MAX:-FLT_MAX;
}

template<class T> inline T pseudo_inverse(const T a) {
  return a?1/a:T();
}

template<class T1,class T2> inline T1 pseudo_divide(const T1& a, const T2& b) {
  return b?a/b:T1();
}

template<class T> inline T robust_harmonic_mean(T a, T b) {
  small_sort(a,b);
  assert(a>=0);
  return b>0?a/(1+a/b)*2:0;
}

template<class T> inline T sinc(const T x) { // sin(x)/x
  return x?sin(x)/x:1;
}

template<class T> inline T one_minus_cos_x_over_x_squared(const T x) { // (1-cos(x))/x^2
  return (T).5*sqr(sinc((T).5*x));
}

template<class T> inline T one_minus_cos_x_over_x(const T x) { // (1-cos(x))/x
  return one_minus_cos_x_over_x_squared(x)*x;
}

template<class T> inline T atan2_y_x_over_y(const T y,const T x) { // atan2(y,x)/y
  return y?atan2(y,x)/y:1;
}

}
