//#####################################################################
// Function Normalize
//#####################################################################
#pragma once

namespace other {

template<class T>
inline typename T::Scalar normalize(T& v)
{return v.normalize();}

template<class T>
inline T normalized(const T& v)
{return v.normalized();}

inline float normalize(float& a)
{float a_save=a;
if(a>=0){a=1;return a_save;}
else{a=-1;return -a_save;}}

inline double normalize(double& a)
{double a_save=a;
if(a>=0){a=1;return a_save;}
else{a=-1;return -a_save;}}

inline float normalized(const float a)
{return a>=0?(float)1:(float)-1;}

inline double normalized(const double a)
{return a>=0?(double)1:(double)-1;}

}
