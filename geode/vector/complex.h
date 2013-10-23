// Complex arithmetic
#pragma once

#include <geode/vector/forward.h>
#include <complex>
namespace geode {

using std::sin;
using std::cos;
using std::arg;
using std::sqrt;
using std::conj;
using std::complex;

template<class T> struct IsScalarBlock<complex<T> >:public IsScalarBlock<T>{};
template<class T> struct IsScalarVectorSpace<complex<T> >:public IsScalarVectorSpace<T>{};
template<class T> struct is_packed_pod<complex<T> >:public is_packed_pod<T>{};

template<class T> static inline T sqr_magnitude(const complex<T>& c) {
  return sqr(c.real())+sqr(c.imag());
}

template<class T> static inline T magnitude(const complex<T>& c) {
  return sqrt(sqr(c.real())+sqr(c.imag()));
}

template<class T> static inline T normalize(complex<T>& c) {
  T mag = magnitude(c);
  if (mag) c *= 1/mag;
  else c = 1;
  return mag;
}

template<class T> static inline complex<T> normalized(const complex<T>& c) {
  complex<T> n = c;
  n.normalize();
  return n;
}

template<class T> static inline complex<T> rotate_left_90(const complex<T>& c) {
  return complex<T>(-c.imag(),c.real());
}

template<class T> static inline complex<T> rotate_right_90(const complex<T>& c) {
  return complex<T>(c.imag(),-c.real());
}

template<class T> static inline complex<T> cis(T theta) {
  return complex<T>(cos(theta),sin(theta));
}

template<class T> static inline T dot(const complex<T>& c1, const complex<T>& c2) {
  return c1.real()*c2.real()+c1.imag()*c2.imag();
}

}
