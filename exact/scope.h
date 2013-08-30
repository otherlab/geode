// Conservative interval arithmetic
#pragma once

// Modified from code by Tyson Brochu, 2011

#include <other/core/math/float-env.h>

namespace other {

// All interval arithmetic must occur within an IntervalScope.  Since the rounding
// mode is local to each thread, care must be taken within multithreaded code.
struct IntervalScope {
  const int previous_mode;

  IntervalScope()
    : previous_mode(get_rounding_mode()) {
   set_rounding_mode(FE_UPWARD);
  }

  ~IntervalScope() {
    set_rounding_mode(previous_mode);
  }

private:
  // Noncopyable
  IntervalScope(const IntervalScope& rhs): previous_mode(0) { OTHER_UNREACHABLE(); };
 IntervalScope& operator=(const IntervalScope& rhs) { OTHER_UNREACHABLE(); };
};

}
