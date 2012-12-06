//#####################################################################
// Function wrap
//#####################################################################
//
// wrap(i,n) adds a multiple of n to i to bring it into the set [0;n)
// i + k*n is in [0,n)
//
//#####################################################################
#pragma once

#include <cmath>
#include <cassert>
namespace other {

using std::fmod;
using std::abs;

static inline int wrap(const int i, const int n) {
  assert(n > 0);
  int r = i % n;
  return r >= 0 ? r : r + n;
}

static inline int wrap(const int i, const int lower, const int upper) {
  int r = i-lower % (upper-lower);
  return r >= 0 ? r+lower : r+upper;
}

static inline float wrap(const float value, const float lower, const float upper) {
  float r = fmod(value-lower,upper-lower);
  return r<0?r+upper:r+lower;
}

static inline double wrap(const double value,const double lower,const double upper) {
  double r = fmod(value-lower,upper-lower);
  return r<0?r+upper:r+lower;
}

}
