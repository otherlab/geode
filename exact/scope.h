// Conservative interval arithmetic
#pragma once

// Modified from code by Tyson Brochu, 2011

#include <other/core/utility/rounding.h>
namespace other {

// All interval arithmetic must occur within an IntervalScope.  Since the rounding
// mode is local to each thread, care must be taken within multithreaded code.
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
