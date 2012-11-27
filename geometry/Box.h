//#####################################################################
// Class Box
//#####################################################################
//
// Box<TV> is (conservatively) exact for set operations, but *inexact*
// for arithmetic.  See core/exact if you need exact operations.
//
//#####################################################################
#pragma once

#include <other/core/geometry/BoxScalar.h>
#include <other/core/geometry/BoxVector.h>
namespace other {

using std::numeric_limits;

#ifdef OTHER_VARIADIC
template<class TV,class... Rest> static inline Box<TV> bounding_box(const TV& p0, const TV& p1, const Rest&... rest) {
  Box<TV> box(p0);
  box.enlarge_nonempty(p1,rest...);
  return box;
}
#else
template<class TV> static inline Box<TV> bounding_box(const TV& p0, const TV& p1) {
  Box<TV> box(p0);
  box.enlarge_nonempty(p1);
  return box;
}
template<class TV> static inline Box<TV> bounding_box(const TV& p0, const TV& p1, const TV& p2) {
  Box<TV> box(p0);
  box.enlarge_nonempty(p1,p2);
  return box;
}
template<class TV> static inline Box<TV> bounding_box(const TV& p0, const TV& p1, const TV& p2, const TV& p3) {
  Box<TV> box(p0);
  box.enlarge_nonempty(p1,p2,p3);
  return box;
}
#endif

template<class TArray> inline Box<typename TArray::Element> bounding_box(const TArray& points) {
  typedef typename TArray::Element T;
  if (!points.size())
    return Box<T>::empty_box();
  Box<T> box(points[0]);
  for (int i=1;i<points.size();i++)
    box.enlarge_nonempty(points[i]);
  return box;
}

template<class TV> static inline Box<TV> operator+(const TV& a, const Box<TV>& b) {
  return Box<TV>(a+b.min,a+b.max);
}

template<class TV> static inline Box<TV> operator-(const TV& a, const Box<TV>& b) {
  return Box<TV>(a-b.max,a-b.min);
}

template<class TV> static inline Box<TV> operator*(const typename ScalarPolicy<TV>::type a, const Box<TV>& box) {
  return box*a;
}

template<class TV> static inline Box<TV> exp(const Box<TV>& x) {
  return Box<TV>(exp(x.min),exp(x.max));
}

template<class TV> inline std::ostream& operator<<(std::ostream& output, const Box<TV>& box) {
  return output<<'['<<box.min<<','<<box.max<<']';
}

template<class TV> inline std::istream& operator>>(std::istream& input,Box<TV>& box) {
  return input>>expect('[')>>box.min>>expect(',')>>box.max>>expect(']');
}

}
