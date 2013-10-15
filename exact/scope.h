// Conservative interval arithmetic.  WARNING: Use with caution (see below)
#pragma once

// Modified from code by Tyson Brochu, 2011

#include <other/core/utility/rounding.h>
namespace other {

// All interval arithmetic must occur within an IntervalScope.  Since the rounding
// mode is local to each thread, care must be taken within multithreaded code.
//
// WARNING: On some compilers (clang!!) there is no way to make rounding mode changes
// completely safe w.r.t. code motion.  The most reliable method is to use IntervalScope
// only at the top of functions, and to mark these functions with OTHER_NEVER_INLINE.
//
struct IntervalScope {
  const int previous_mode;

  IntervalScope()
    : previous_mode(fegetround()) {
    fesetround(FE_UPWARD);
  }

  ~IntervalScope() {
    fesetround(previous_mode);
  }

private:
  // Noncopyable
  IntervalScope(const IntervalScope& rhs);
  IntervalScope& operator=(const IntervalScope& rhs);
};

}
