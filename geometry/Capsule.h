//#####################################################################
// Class Capsule
//#####################################################################
#pragma once

#include <other/core/geometry/Segment2d.h>
#include <other/core/geometry/Segment3d.h>
#include <other/core/vector/magnitude.h>
namespace other {

template<class TV> class Capsule {
  typedef typename TV::Scalar T;
  enum Workaround {d=TV::m};
public:
  typedef TV VectorT;

  Segment<TV> segment;
  T radius;

  Capsule(const TV& x0, const TV& x1, const T radius)
    : segment(x0,x1), radius(radius) {}

  TV normal(const TV& X) const {
    TV N = X-segment.closest_point(X);
    T sphi = magnitude(N);
    return sphi?N/sphi:(segment.x1-segment.x0).unit_orthogonal_vector();
  }

  bool inside(const TV& X,const T thickness_over_two) const {
    return sqr_magnitude(X-segment.closest_point(X)) <= sqr(radius-thickness_over_two);
  }

  bool lazy_inside(const TV& X) const {
    return sqr_magnitude(X-segment.closest_point(X)) <= sqr(radius);
  }

  TV surface(const TV& X) const {
    TV C = segment.closest_point(X);
    TV N = X-C;
    T sphi = magnitude(N);
    return C + (sphi?radius/sphi*N:radius*(segment.x1-segment.x0).unit_orthogonal_vector());
  }

  T phi(const TV& X) const {
    return magnitude(X-segment.closest_point(X))-radius;
  }

  T volume() const {
    return (T)unit_sphere_size<d  >::value*pow<d  >(radius)
          +(T)unit_sphere_size<d-1>::value*pow<d-1>(radius)*segment.length();
  }

  Box<TV> bounding_box() const {
    return segment.bounding_box().thickened(radius);
  }

  string repr() const {
    return format("Capsule(%s,%s,%s)",tuple_repr(segment.x0),tuple_repr(segment.x1),other::repr(radius));
  }
};

}
