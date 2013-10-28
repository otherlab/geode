#pragma once

#include <geode/python/forward.h>
namespace geode {

template<class T>
inline T lerp(real a, T const &x, T const &y) {
  return (1 - a) * x + a * y;
}

template<class T>
inline T lerp(real a, real min, real max, T const &x, T const &y) {
  a = (a-min)/(max-min);
  return lerp(a, x, y);
}

}
