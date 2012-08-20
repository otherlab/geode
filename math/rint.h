//#####################################################################
// Function rint
//#####################################################################
#pragma once

#ifdef Win32
#include <math.h>
#else
#include <cmath>
#endif
namespace other {

#ifdef Win32
inline float rint(const float x){return floorf(x+(x>0?.5f:-.5f));}
inline double rint(const double x){return floor(x+(x>0?.5:-.5));}
#else
inline float rint(const float x){return ::rintf(x);}
inline double rint(const double x){return ::rint(x);}
#endif

}
