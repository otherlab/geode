#pragma once

#include <cmath>
namespace other {
  
#ifdef _WIN32
  
template<class T> static inline bool isfinite(T x) {
  return _finite(x)!=0;
}
  
#else
  
#undef isfinite
using std::isfinite;
  
#endif
}
