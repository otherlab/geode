//#####################################################################
// Function cbrt
//#####################################################################
//
// Apparently cbrt is standard C but not standard C++.
//
//#####################################################################
#pragma once

#ifdef _WIN32
#include <math.h>
#else
#include <cmath>
#endif
namespace other {

#ifdef _WIN32
inline float cbrt(const float a){return ::pow(a,(float)(1/3.));}
inline double cbrt(const double a){return ::pow(a,(double)(1/3.));}
#else
inline float cbrt(const float a){return ::cbrtf(a);}
inline double cbrt(const double a){return ::cbrt(a);}
#endif

}
