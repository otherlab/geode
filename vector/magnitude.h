//#####################################################################
// Function Magnitude
//#####################################################################
#pragma once

#include <cmath>
#include <cstdlib>
namespace other {

using ::std::abs;

template<class TV>
inline typename TV::Scalar magnitude(const TV& v)
{return v.magnitude();}

template<class TV>
inline typename TV::Scalar sqr_magnitude(const TV& v)
{return v.sqr_magnitude();}

inline int magnitude(const int a)
{return abs(a);}

inline float magnitude(const float a)
{return abs(a);}

inline double magnitude(const double a)
{return abs(a);}

inline float sqr_magnitude(const float a)
{return a*a;}

inline double sqr_magnitude(const double a)
{return a*a;}

}
