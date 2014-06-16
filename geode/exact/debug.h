// Optional expensive debugging for robust geometric computation
// This file should be included only in implementation .cpp files.
#pragma once

#include <geode/exact/config.h>
#include <geode/exact/Interval.h>
#include <geode/utility/str.h>
namespace geode {

// Some predicate tests can be accelerated with interval arithmetic using simpler
// formulas than those required for full perturbed robustness.  If the following
// flag is true, we always run both fast and slow versions and compare result.
// IMPORTANT: This is a much stronger test than pure unit tests, and should be run
// for any nontrivial changes to exact algorithms.
#define CHECK 0

// Run a fast interval check, and fall back to a slower exact check if it fails.
// If CHECK is true, do both and validate.
#if !CHECK
#ifdef __GNUC__
// In gcc, we can define a clean macro that evaluates its arguments at most once time.
#define FILTER(fast,...) ({ \
  const int _s = weak_sign(fast); \
  _s ? _s>0 : (__VA_ARGS__); })
#else
// Warning: we lack gcc, the argument must be evaluated multiple times.  Hopefully CSE will do its work.
#define FILTER(fast,...) \
  (  certainly_positive(fast) ? true \
   : certainly_negative(fast) ? false \
   : (__VA_ARGS__))
#endif
#else
// In check mode, always do both.
GEODE_UNUSED static bool filter_helper(const Interval fast, const bool slow, const int line) {
  GEODE_WARNING("Expensive consistency checking enabled");
  const int sign = weak_sign(fast);
  if (sign && (sign>0)!=slow)
    throw AssertionError(format("circle_csg: Consistency check failed on line %d, interval %s, slow sign %d",
                                line,str(fast),slow?1:-1));
  return slow;
}
#define FILTER(fast,...) filter_helper(fast,__VA_ARGS__,__LINE__)
#endif 

}
