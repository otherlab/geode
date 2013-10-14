// Floating point rounding mode control
#pragma once

#ifndef _WIN32
#include <fenv.h>
#else
#include <float.h>
#include <cassert>
#endif
namespace other {

// We use the native API on posix, or emulate it on Windows.
// These functions should not be called in performance critical loops.

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

// Clang optimizations don't respect fesetround, so trick it into avoiding code
// motion around such calls.  For safety, do this for all other compilers as well.
// This is a cludge, but a necessary one due to clang's stupidity.
static inline void safe_fesetround(int mode) {
  OTHER_ASSERT(!fesetround(mode));
}

}
