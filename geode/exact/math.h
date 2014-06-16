// Geometric and linear algebraic utility routines for use in exact predicates.
// These are more general than their equivalents in geode/vector: they can operate on tuples with varying precision.
#pragma once

#include <geode/utility/config.h>
#include <geode/utility/type_traits.h>
namespace geode {

template<class U,class V> static auto edot(const U& u, const V& v)
  -> typename enable_if_c<U::m==2 && V::m==2, decltype(u.x*v.x+u.y*v.y)>::type {
  return u.x*v.x+u.y*v.y;
}

template<class U,class V> static auto edot(const U& u, const V& v)
  -> typename enable_if_c<U::m==3 && V::m==3, decltype(u.x*v.x+u.y*v.y+u.z*v.z)>::type {
  return u.x*v.x+u.y*v.y+u.z*v.z;
}

template<class U,class V> static auto ecross(const U& u, const V& v)
  -> typename enable_if_c<U::m==3 && V::m==3,Vector<decltype(u.x*v.y),3>>::type {
  return Vector<decltype(u.x*v.y),3>(u.y*v.z-u.z*v.y,
                                     u.z*v.x-u.x*v.z,
                                     u.x*v.y-u.y*v.x);
}

template<class U> static inline auto enormal(const U& u0, const U& u1, const U& u2)
  -> decltype(ecross(u1-u0,u2-u0)) {
  return ecross(u1-u0,u2-u0);
}

template<class U> static auto esqr_magnitude(const U& u)
  -> typename enable_if_c<U::m==2,decltype(sqr(u.x)+sqr(u.y))>::type {
  return sqr(u.x)+sqr(u.y);
}

template<class U> static auto esqr_magnitude(const U& u)
  -> typename enable_if_c<U::m==3,decltype(sqr(u.x)+sqr(u.y)+sqr(u.z))>::type {
  return sqr(u.x)+sqr(u.y)+sqr(u.z);
}

template<class U,class V> static auto edet(const U& u, const V& v)
  -> typename enable_if_c<U::m==2 && V::m==2,decltype(u.x*v.y-u.y*v.x)>::type {
  return u.x*v.y-u.y*v.x;
}

template<class U,class V,class W> static auto edet(const U& u, const V& v, const W& w)
  -> typename enable_if_c<U::m==3 && V::m==3 && W::m==3,decltype(u.x*(v.y*w.z-v.z*w.y)+v.x*(w.y*u.z-w.z*u.y)+w.x*(u.y*v.z-u.z*v.y))>::type {
  return u.x*(v.y*w.z-v.z*w.y)+v.x*(w.y*u.z-w.z*u.y)+w.x*(u.y*v.z-u.z*v.y);
}

template<class U,class V> static auto emul(const U& s, const Vector<V,2>& v)
  -> Vector<decltype(s*v.x),2> {
  return Vector<decltype(s*v.x),2>(s*v.x,s*v.y);
}

template<class U,class V> static auto emul(const U& s, const Vector<V,3>& v)
  -> Vector<decltype(s*v.x),3> {
  return Vector<decltype(s*v.x),3>(s*v.x,s*v.y,s*v.z);
}

template<class T> static auto esqr(const Vector<T,2>& u)
  -> decltype(vec(sqr(u.x),sqr(u.y))) {
  return vec(sqr(u.x),sqr(u.y));
}

}
