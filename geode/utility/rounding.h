// Floating point rounding mode control
#pragma once

#ifndef _WIN32
#include <fenv.h>
#else
#include <float.h>
#include <cassert>
#endif
namespace geode {

// We use the native API on posix, or emulate it on Windows.
// These functions should not be called in performance critical loops.

// Be careful when using fesetround as Clang can 'optimize' your code by moving
// operations past the fesetround call and has no safe mechanism to prevent this.

#ifdef _WIN32

// Teach Windows that we care about floating point rounding modes
#pragma fenv_access (on)

enum {
  FE_TONEAREST  = _RC_NEAR,
  FE_UPWARD     = _RC_UP,
  FE_DOWNWARD   = _RC_DOWN,
  FE_TOWARDZERO = _RC_CHOP
};

static inline int fegetround() {
  return _controlfp(0,0) & _MCW_RC;
}

static inline int fesetround(int mode) {
  assert((mode&_MCW_RC)==mode);
  _controlfp(mode,_MCW_RC);
  return 0; // Always indicate success
}

#endif

}
