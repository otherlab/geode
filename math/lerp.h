#pragma once

template<class T>
inline T lerp(double a, T const &x, T const &y) {
  return (1 - a) * x + a * y;
}

template<class T>
inline T lerp(double a, double min, double max, T const &x, T const &y) {
  a = (a-min)/(max-min);
  return lerp(a, x, y);
}
