#pragma once

#include <other/core/geometry/Box.h>
#include <other/core/geometry/Plane.h>
#include <other/core/math/constants.h>
namespace other {

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

  Cylinder(TV x0, TV x1, T radius);

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

ostream& operator<<(ostream& output, const Cylinder& cylinder) OTHER_EXPORT;
PyObject* to_python(const Cylinder& cylinder) OTHER_EXPORT;
template<> struct FromPython<Cylinder>{ static Cylinder convert(PyObject* object) OTHER_EXPORT; };

}
