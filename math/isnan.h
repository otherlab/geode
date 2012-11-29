//#####################################################################
// Function isnan
//#####################################################################
#pragma once

#include <cmath>
namespace other {

#ifdef _WIN32

template<class T> static inline bool isnan(T x) {
  return _isnan(x)!=0;
}

#else

using std::isnan;

#endif
}
