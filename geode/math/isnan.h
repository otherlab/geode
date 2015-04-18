//#####################################################################
// Function isnan
//#####################################################################
#pragma once

#include <cmath>
namespace geode {

#ifndef __GNUC__

template<class T> static inline bool isnan(T x) {
  return _isnan(x)!=0;
}

#else

using std::isnan;

#endif
}
