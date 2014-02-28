// Segments
#pragma once

#include <geode/array/forward.h>
#include <geode/geometry/Box.h>
#include <geode/vector/Vector.h>
#include <geode/vector/normalize.h>
namespace geode {

template<class T> static inline Vector<T,2> normal(const Vector<T,2>& x0, const Vector<T,2>& x1) {
  return rotate_right_90(normalized(x1-x0));
}

template<class TA> static inline typename EnableForSize<2,TA,typename TA::Element>::type normal(const TA& X) {
  return normal(X[0],X[1]);
}

template<class TV> class Segment {
  typedef typename TV::Scalar T;
public:
  TV x0,x1;

  Segment() {}

  Segment(const TV& x0, const TV& x1)
    : x0(x0), x1(x1) {}

  template<class TA> explicit Segment(const TA& X)
    : x0(X[0]), x1(X[1]) {
    static_assert(TA::m==2,"");
  }

  const TV& X(const int i) const {
    assert(unsigned(i)<2);return (&x0)[i];
  }

  TV& X(const int i) {
    assert(unsigned(i)<2);return (&x0)[i];
  }

  T length() const {
    return magnitude(x1-x0);
  }

  T size() const {
    return length();
  }

  static T size(const TV& x0, const TV& x1) {
    return magnitude(x1-x0);
  }

  template<class TA> static T size(const TA& X) {
    static_assert(TA::m==2,"");
    return magnitude(X(1)-X(0));
  }

  template<class TA> static T signed_size(const TA& X) {
    return size(X);
  }

  TV vector() const {
    return x1-x0;
  }

  Box<TV> bounding_box() const {
    return geode::bounding_box(x0,x1);
  }

  TV center() const {
    return (T).5*(x0+x1);
  }

  TV normal() const {
    return geode::normal(x0,x1);
  }

  TV interpolate(const T t) const {
    return x0+t*(x1-x0);
  }

  // Closest point and weight
  Tuple<TV,T> closest_point(const TV& p) const {
    return segment_closest_point(*this,p);
  }

  T distance(const TV& p) const {
    return segment_point_distance(*this,p);
  }

  // For templatization purposes
  static T min_weight(const T w) {
    return min(w,1-w);
  }

  bool intersection(Ray<Vector<T,2>>& ray,const T half_thickness) const {
    return segment_ray_intersection(*this,ray,half_thickness);
  }
};

template<class TV> GEODE_CORE_EXPORT PyObject* to_python(const Segment<TV>& seg);

template<class TV> static inline Segment<TV> simplex(const TV& x0, const TV& x1) {
  return Segment<TV>(x0,x1); 
}

template<class T,class TV> static inline TV interpolate(const T alpha, const TV& x0, const TV& x1) {
  return x0+alpha*(x1-x0);
}

template<class T,class TA> static inline typename EnableForSize<2,TA,typename TA::Element>::type interpolate(const T alpha, const TA& X) {
  const auto &x0 = X(0), &x1 = X(1);
  return x0+alpha*(x1-x0);
}

template<class TV> GEODE_CORE_EXPORT real interpolation_fraction(const Segment<TV>& s, const TV p);
template<class TV> GEODE_CORE_EXPORT real clamped_interpolation_fraction(const Segment<TV>& s, const TV p);

template<class TV> static inline real barycentric_coordinates(const Segment<TV>& s, const TV p) {
  return interpolation_fraction(s,p);
}
template<class TV> static inline real clamped_barycentric_coordinates(const Segment<TV>& s, const TV p) {
  return clamped_interpolation_fraction(s,p);
}

template<class TV> std::ostream& operator<<(std::ostream& output, Segment<TV> const &s) {
  return output<<'['<<s.x0<<','<<s.x1<<']';
}

// The following routines are stable even for degenerate, parallel, or nearly parallel segments.  However, the meaning of stable is subtle
// where the function being computed is not continuous (line-line distance, any normal, etc.).  Invariants guaranteed to hold for nondegenerate
// configurations should always hold, however: all normals will have unit length, all weights will be in [0,1], etc.
//
// Normals always point from first to second argument.

/********* Segment / Point *********/

template<int d> GEODE_CORE_EXPORT real segment_point_distance(Segment<Vector<real,d>> s, Vector<real,d> p);
template<int d> GEODE_CORE_EXPORT real segment_point_sqr_distance(Segment<Vector<real,d>> s, Vector<real,d> p);

// Snap a point to the closest point on a segment, and return (closest,weight).
template<int d> GEODE_CORE_EXPORT Tuple<Vector<real,d>,real> segment_closest_point(Segment<Vector<real,d>> s, Vector<real,d> p);

// Distance,normal,weight for a point,segment pair.  Note that the normal cannot be stably computed from segment_closest_point.
template<int d> GEODE_CORE_EXPORT Tuple<real,Vector<real,d>,real> segment_point_distance_and_normal(Segment<Vector<real,d>> s, Vector<real,d> p);

/********* Segment / Segment *********/

template<int d> GEODE_CORE_EXPORT real segment_segment_distance(Segment<Vector<real,d>> s0, Segment<Vector<real,d>> s1);

// Compute distance,normal,weights.  normal points from s0 to s1, and weights are for (s0,s1).  This function is stable even for nearly parallel segments,
// and in particular returns a valid normal even if the distance is zero.
template<int d> GEODE_CORE_EXPORT Tuple<real,Vector<real,3>,Vector<real,2>> segment_segment_distance_and_normal(Segment<Vector<real,d>> s0, Segment<Vector<real,d>> s1);

/********* Segment / Ray *********/

// Segment is lengthened at each end by half_thickness
GEODE_CORE_EXPORT bool segment_ray_intersection(const Segment<Vector<real,2>>& seg, Ray<Vector<real,2>>& ray, const real half_thickness);

/********* Point / Line *********/

template<int d> GEODE_CORE_EXPORT real line_point_distance(Segment<Vector<real,d>> s, Vector<real,d> p);

template<int d> GEODE_CORE_EXPORT Tuple<Vector<real,d>,real> line_closest_point(Segment<Vector<real,d>> s, Vector<real,d> p);

// Distance,normal,weight
GEODE_CORE_EXPORT Tuple<real,Vector<real,3>,real> line_point_distance_and_normal(Segment<Vector<real,3>> s, Vector<real,3> p);

/********* Line / Line *********/

// Same as segment_segment_distance_and_normal, but computes result for infinite lines.
// For exactly or nearly parallel lines, the distance will be between 0 and the actual distance+epsilon.
// This is intentional and correct: the result will be backward stable since a slight change can produce any such value.
GEODE_CORE_EXPORT Tuple<real,Vector<real,3>,Vector<real,2>> line_line_distance_and_normal(Segment<Vector<real,3>> s0, Segment<Vector<real,3>> s1);

}
