//#####################################################################
// Class Box<T>
//#####################################################################
//
// A range of scalars.  Set operations are exact, since they're just min/max
// but arithmetic operations are *not* exact.  See core/exact if you need
// exact operations.
//
//#####################################################################
#pragma once

#include <other/core/geometry/forward.h>
#include <other/core/vector/ScalarPolicy.h>
#include <other/core/math/clamp.h>
#include <other/core/math/max.h>
#include <other/core/math/min.h>
#include <other/core/math/maxabs.h>
#include <other/core/math/Zero.h>
#include <other/core/python/forward.h>
#include <other/core/utility/STATIC_ASSERT_SAME.h>
#include <boost/mpl/assert.hpp>
#include <cassert>
#include <limits>
#include <cfloat>
namespace other {

using std::numeric_limits;

template<class T> OTHER_CORE_EXPORT PyObject* to_python(const Box<T>& self);
template<class T> struct FromPython<Box<T> >{OTHER_CORE_EXPORT static Box<T> convert(PyObject* object);};

template<class T> class Box {
public:
  BOOST_MPL_ASSERT((IsScalar<T>)); // See BoxVector.h for the Vector version
  typedef T Scalar;

  T min,max;

  Box()
    : min(numeric_limits<T>::max()), max(-numeric_limits<T>::max()) {}

  Box(const T value)
    : min(value), max(value) {}

  Box(const T min, const T max)
    : min(min), max(max) {}

  template<class T2> explicit Box(const Box<T2>& box)
    : min(T(box.min)), max(T(box.max)) {}

  static Box unit_box() {
    return Box((T)0,(T)1);
  }

  static Box zero_box() {
    return Box(0);
  }

  static Box empty_box() {
    return Box();
  }

  static Box full_box() {
    return Box(-numeric_limits<T>::max(),numeric_limits<T>::max());
  }

  bool empty() const {
    return min>max;
  }

  bool operator==(const Box& r) const {
    return min==r.min && max==r.max;
  }

  bool operator!=(const Box& r) const {
    return !(*this==r);
  }

  Box operator-() const {
    return Box(-max,-min);
  }

  Box& operator+=(const Box& r) {
    min+=r.min;max+=r.max;return *this;
  }

  Box& operator-=(const Box& r) {
    min-=r.max;max-=r.min;return *this;
  }

  Box operator+(const Box& r) const {
    return Box(min+r.min,max+r.max);
  }

  Box operator-(const Box& r) const {
    return Box(min-r.max,max-r.min);
  }

  Box operator*(const T a) const {
    return a>=0?Box(min*a,max*a):Box(max*a,min*a);
  }

  Box& operator*=(const T a) {
    return *this = *this*a;
  }

  Box operator/(const T a) const {
    assert(a!=0);return *this*inverse(a);
  }

  Box& operator/=(const T a) {
    return *this=*this/a;
  }

  T center() const {
    return (T).5*(min+max);
  }

  T size() const {
    return max-min;
  }

  T minabs() const {
    return min>0?min:max<0?-max:0;
  }

  T maxabs() const {
    return other::maxabs(min,max);
  }

  void enlarge(const T& point) {
    min = other::min(min,point);
    max = other::max(max,point);
  }

  void enlarge_nonempty(const T& point) {
    assert(!empty());
    if (point<min) min = point;
    else if (point>max) max = point;
  }

  void enlarge_nonempty(const T& p1, const T& p2) {
    enlarge_nonempty(p1);
    enlarge_nonempty(p2);
  }

  void enlarge_nonempty(const T& p1, const T& p2, const T& p3) {
    enlarge_nonempty(p1);enlarge_nonempty(p2);enlarge_nonempty(p3);}

  template<class TArray> void enlarge_nonempty(const TArray& points) {
    STATIC_ASSERT_SAME(typename TArray::Element,T);
    for (int i=0;i<points.size();i++)
      enlarge_nonempty(points[i]);
  }

  void enlarge(const Box& box) {
    min = other::min(min,box.min);
    max = other::max(max,box.max);
  }

  void change_size(const T delta) {
    min-=delta;max+=delta;
  }

  Box thickened(const T half_thickness) const {
    return Box(min-half_thickness,max+half_thickness);
  }

  static Box combine(const Box& box1, const Box& box2) {
    return Box(other::min(box1.min,box2.min),other::max(box1.max,box2.max));
  }

  static Box intersect(const Box& box1,const Box& box2) {
    return Box(other::max(box1.min,box2.min),other::min(box1.max,box2.max));
  }

  void scale_about_center(const T factor) {
    T center = (T).5*(min+max),
      half_length = factor*(T).5*(max-min);
    min = center-half_length;
    max = center+half_length;
  }

  bool lazy_inside(const T& location) const {
    return min<=location && location<=max;
  }

  bool lazy_inside_half_open(const T& location) const {
    return min<=location && location<max;
  }

  bool inside(const T& location, const T half_thickness) const {
    return thickened(-half_thickness).lazy_inside(location);
  }

  bool inside(const T& location, const Zero half_thickness) const {
    return lazy_inside(location);
  }

  bool lazy_outside(const T& location) const {
    return !lazy_inside(location);
  }

  bool outside(const T& location, const T half_thickness) const {
    return thickened(half_thickness).lazy_outside(location);
  }

  bool outside(const T& location, const Zero half_thickness) const {
    return lazy_outside(location);
  }

  bool boundary(const T& location, const T half_thickness) const {
    bool strict_inside = min+half_thickness<location && location<max-half_thickness;
    return !strict_inside && !outside(location,half_thickness);
  }

  T clamp(const T& location) const {
    return other::clamp(location,min,max);
  }

  bool contains(const Box& box) const {
    return min<=box.min && box.max<=max;
  }

  bool lazy_intersects(const Box& box) const {
    return min<=box.max && box.min<=max;
  }

  bool intersects(const Box& box, const T half_thickness) const {
    return thickened(half_thickness).lazy_intersects(box);
  }

  bool intersects(const Box& box, const Zero half_thickness) const {
    return lazy_intersects(box);
  }

  bool intersects(const Box& box) const {
    return lazy_intersects(box);
  }

  T signed_distance(const T& X) const {
    return abs(X-center())-(T).5*size();
  }
};

template<class T> static inline Box<T> sqr(const Box<T>& x) {
  T smin = sqr(x.min), smax = sqr(x.max);
  return x.min>=0?Box<T>(smin,smax):x.max<=0?Box<T>(smax,smin):Box<T>(0,max(smin,smax));
}

}
