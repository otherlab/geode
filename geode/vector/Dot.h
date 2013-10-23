//#####################################################################
// Function Dot
//#####################################################################
#pragma once

namespace geode {

template<class T,int d> class Vector;

static inline float dot(const float a1,const float a2)
{return a1*a2;}

static inline double dot(const double a1,const double a2)
{return a1*a2;}

template<class T,int d>
static inline double dot_double_precision(const Vector<T,d>& v1,const Vector<T,d>& v2)
{return dot(v1,v2);}

static inline double dot_double_precision(const float a1,const float a2)
{return a1*a2;}

static inline double dot_double_precision(const double a1,const double a2)
{return a1*a2;}

}
