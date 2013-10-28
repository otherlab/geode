//#####################################################################
// Function pow
//#####################################################################
//
// pow is slow
//
//#####################################################################
#pragma once

#include <geode/math/cbrt.h>
#include <cmath>
namespace geode {

using ::std::sqrt;

template<class T,int numerator,int denominator=1> struct PowHelper;

template<class T> struct PowHelper<T,-3>{static T pow(const T a){return 1/(a*a*a);}};
template<class T> struct PowHelper<T,-2>{static T pow(const T a){return 1/(a*a);}};
template<class T> struct PowHelper<T,-1>{static T pow(const T a){return 1/a;}};
template<class T> struct PowHelper<T,0>{static T pow(const T a){return 1;}};
template<class T> struct PowHelper<T,1,2>{static T pow(const T a){return sqrt(a);}};
template<class T> struct PowHelper<T,1,3>{static T pow(const T a){return cbrt(a);}};
template<class T> struct PowHelper<T,1>{static T pow(const T a){return a;}};
template<class T> struct PowHelper<T,2>{static T pow(const T a){return a*a;}};
template<class T> struct PowHelper<T,3>{static T pow(const T a){return a*a*a;}};

template<int numerator,class T> inline T pow(const T a) {
  return PowHelper<T,numerator,1>::pow(a);
}

template<int numerator,int denominator,class T> inline T pow(const T a) {
  return PowHelper<T,numerator,denominator>::pow(a);
}

template<int a,int p> struct Pow;
template<int a> struct Pow<a,0>{static const int value = 1;};
template<int a> struct Pow<a,1>{static const int value = a;};
template<int a> struct Pow<a,2>{static const int value = a*a;};
template<int a> struct Pow<a,3>{static const int value = a*a*a;};
template<int a> struct Pow<a,4>{static const int value = a*a*a*a;};
template<int a> struct Pow<a,5>{static const int value = a*a*a*a*a;};

}
