#pragma once

#ifdef WIN32
#include <float.h>
#else
#include <fenv.h>
#endif

namespace other {

// missing fenv.h functions for windows
#ifndef WIN32

inline int get_rounding_mode() {
  return fegetround();
}

inline void set_rounding_mode(int mode) {
  fesetround(mode);
}

#else

inline int get_rounding_mode() {
  return _controlfp(0,0);
}

enum {
  FE_TONEAREST = _RC_NEAR,
  FE_TOWARDZERO = _RC_CHOP,
  FE_UPWARD = _RC_UP,
  FE_DOWNWARD = _RC_DOWN,
};

inline int set_rounding_mode(int mode) {
  return _controlfp(mode, _MCW_RC);
}

#endif

}
