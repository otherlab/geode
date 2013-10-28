//#####################################################################
// Class Capsule
//#####################################################################
#pragma once

#include <geode/geometry/Segment.h>
#include <geode/structure/Tuple.h>
#include <geode/vector/magnitude.h>
namespace geode {

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
    return segment_point_distance_and_normal(segment,X).y;
  }

  bool inside(const TV& X, const T half_thickness) const {
    return segment_point_distance(segment,X) <= radius-half_thickness;
  }

  bool lazy_inside(const TV& X) const {
    return segment_point_distance(segment,X) <= radius;
  }

  TV surface(const TV& X) const {
    const auto r = segment_point_distance_and_normal(segment,X);
    return segment.interpolate(r.z)+radius*r.y;
  }

  T phi(const TV& X) const {
    return segment_point_distance(segment,X)-radius;
  }

  T volume() const {
    return (T)unit_sphere_size<d  >::value*pow<d  >(radius)
          +(T)unit_sphere_size<d-1>::value*pow<d-1>(radius)*segment.length();
  }

  Box<TV> bounding_box() const {
    return segment.bounding_box().thickened(radius);
  }

  string repr() const {
    return format("Capsule(%s,%s,%s)",tuple_repr(segment.x0),tuple_repr(segment.x1),geode::repr(radius));
  }
};

}
