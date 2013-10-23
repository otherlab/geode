//#####################################################################
// Function copysign
//#####################################################################
#pragma once

#include <stdint.h>
namespace geode {

// The normal copysign functions don't always inline, so we make our own

static inline float copysign(float mag, float sign) {
  uint32_t mask = 1<<31;
  union {float x;uint32_t i;} mag_, sign_;
  mag_.x = mag;
  sign_.x = sign;
  mag_.i = (mag_.i&~mask)|(sign_.i&mask);
  return mag_.x;
}

static inline double copysign(double mag, double sign) {
  uint64_t mask = (uint64_t)1<<63;
  union {double x;uint64_t i;} mag_, sign_;
  mag_.x = mag;
  sign_.x = sign;
  mag_.i = (mag_.i&~mask)|(sign_.i&mask);
  return mag_.x;
}

}
