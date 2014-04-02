#pragma once

#include <geode/geometry/Box.h>
#include <geode/geometry/Plane.h>
#include <geode/math/constants.h>
namespace geode {

using std::ostream;

class Cylinder
{
public:
  typedef real T;
  typedef Vector<T,3> TV;
  enum {d = 3};
  typedef TV VectorT;

  Plane<T> base; // normal points through the cylinder
  T radius, height;

  Cylinder()
    : radius(1), height(1) {}

  Cylinder(const Plane<T>& base, T radius, T height)
    : base(base), radius(radius), height(height) {}

  GEODE_CORE_EXPORT Cylinder(TV x0, TV x1, T radius);

  bool operator==(const Cylinder& other) const;
  TV normal(const TV& X) const;
  bool inside(const TV& X,const T half_thickness) const;
  bool lazy_inside(const TV& X) const;
  TV surface(const TV& X) const;
  T phi(const TV& X) const;
  T volume() const;
  Box<TV> bounding_box() const;
  Vector<T,d-1> principal_curvatures(const TV& X) const;
  bool lazy_intersects(const Box<TV>& box) const;
  string repr() const;
};

GEODE_CORE_EXPORT ostream& operator<<(ostream& output, const Cylinder& cylinder);

}
