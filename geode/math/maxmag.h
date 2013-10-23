//#####################################################################
// Function maxmag
//#####################################################################
//
// finds the maximum value in magnitude and returns it with the sign
//
//#####################################################################
#pragma once

#include <cmath>
namespace geode {

using std::abs;

template<class T> inline T maxmag(const T a,const T b) {
  return abs(a)>abs(b)?a:b;
}

template<class T> inline T maxmag(const T a,const T b,const T c) {
  return maxmag(a,maxmag(b,c));
}

template<class T> inline T maxmag(const T a,const T b,const T c,const T d) {
  return maxmag(a,maxmag(b,c,d));
}

template<class T> inline T maxmag(const T a,const T b,const T c,const T d,const T e) {
  return maxmag(a,maxmag(b,c,d,e));
}

template<class T> inline T maxmag(const T a,const T b,const T c,const T d,const T e,const T f) {
  return maxmag(a,maxmag(b,c,d,e,f));
}

}
