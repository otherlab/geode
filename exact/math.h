// Geometric and linear algebraic utility routines for use in exact predicates.
// These are more general than their equivalents in core/vector: they can operate on tuples with varying precision.
#pragma once

#include <other/core/utility/config.h>
#include <boost/utility/enable_if.hpp>
namespace other {

template<class U,class V> static auto edot(const U& u, const V& v)
  -> typename boost::enable_if_c<U::m==2 && V::m==2, decltype(u.x*v.x+u.y*v.y)>::type {
  return u.x*v.x+u.y*v.y;
}

template<class U,class V> static auto edot(const U& u, const V& v)
  -> typename boost::enable_if_c<U::m==3 && V::m==3, decltype(u.x*v.x+u.y*v.y+u.z*v.z)>::type {
  return u.x*v.x+u.y*v.y+u.z*v.z;
}

template<class U> static auto esqr_magnitude(const U& u)
  -> typename boost::enable_if_c<U::m==2,decltype(sqr(u.x)+sqr(u.y))>::type {
  return sqr(u.x)+sqr(u.y);
}

template<class U> static auto esqr_magnitude(const U& u)
  -> typename boost::enable_if_c<U::m==3,decltype(sqr(u.x)+sqr(u.y)+sqr(u.z))>::type {
  return sqr(u.x)+sqr(u.y)+sqr(u.z);
}

template<class U,class V> static auto edet(const U& u, const V& v)
  -> typename boost::enable_if_c<U::m==2 && V::m==2,decltype(u.x*v.y-u.y*v.x)>::type {
  return u.x*v.y-u.y*v.x;
}

template<class U,class V,class W> static auto edet(const U& u, const V& v, const W& w)
  -> typename boost::enable_if_c<U::m==3 && V::m==3 && W::m==3,decltype(u.x*(v.y*w.z-v.z*w.y)+v.x*(w.y*u.z-w.z*u.y)+w.x*(u.y*v.z-u.z*v.y))>::type {
  return u.x*(v.y*w.z-v.z*w.y)+v.x*(w.y*u.z-w.z*u.y)+w.x*(u.y*v.z-u.z*v.y);
}

template<class U,class V> static auto emul(const U& s, const Vector<V,2>& v)
  -> Vector<decltype(s*v.x),2> {
  return Vector<decltype(s*v.x),2>(s*v.x,s*v.y);
}

template<class T> static auto esqr(const Vector<T,2>& u)
  -> decltype(vec(sqr(u.x),sqr(u.y))) {
  return vec(sqr(u.x),sqr(u.y));
}

}
