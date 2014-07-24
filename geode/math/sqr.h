// Square a number
#pragma once

#include <geode/math/copysign.h>
namespace geode {

template<class T> static inline auto sqr(const T& a)
  -> decltype(a*a) {
  return a*a;
}

// A monotonic version of sqr
template<class T> static inline auto sign_sqr(const T& a)
  -> decltype(a*a) {
  return copysign(a*a,a);
}

}
