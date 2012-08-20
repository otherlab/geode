//#####################################################################
// Function isnan
//#####################################################################
#pragma once

#include <cmath>
namespace other {

#ifdef Win32

template<class T> static inline bool isnan(T x) {
  return _isnan(x);
}

#else

using std::isnan;

#endif
}
