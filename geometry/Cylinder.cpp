#include <other/core/geometry/Cylinder.h>
#include <other/core/vector/magnitude.h>
#include <other/core/vector/normalize.h>
namespace other {

typedef real T;
typedef Vector<T,3> TV;

Cylinder::Cylinder(TV x0, TV x1, T radius)
  : radius(radius) {
  base.x0 = x0;
  base.n = x1-x0;
  height = base.n.normalize();
}

bool Cylinder::operator==(const Cylinder& other) const {
  return base.x0==other.base.x0 && base.n==other.base.n && radius==other.radius && height==other.height;
}

TV Cylinder::normal(const TV& X) const {
  const TV v = X - base.x0;
  const T h0 = -dot(v,base.n),
          h1 = -h0-height,
          h = max(h0,h1),
          sh = h1>h0?1:-1;
  TV dr = v+h0*base.n;
  const T r = magnitude(dr);
  dr = r?dr/r:base.n.unit_orthogonal_vector();
  const T rp = r-radius;
  if (rp>0 && h>0) { // Outside
    T mag = magnitude(vec(rp,h));
    return rp/mag*dr+sh*h/mag*base.n;
  } else if (rp > h) // Closest to infinite cylinder
    return dr;
  return sh*base.n; // Closest to an end cap
}

bool Cylinder::inside(const TV& X,const T half_thickness) const {
  const TV v = X - base.x0;
  const T h = dot(v,base.n);
  return h>=-half_thickness && h<=height+half_thickness && sqr_magnitude(v-h*base.n)<=sqr(radius+half_thickness);
}

bool Cylinder::lazy_inside(const TV& X) const {
  const TV v = X - base.x0;
  const T h = dot(v,base.n);
  return h>=0 && h<=height && sqr_magnitude(v-h*base.n)<=sqr(radius);
}

TV Cylinder::surface(const TV& X) const {
  const TV v = X - base.x0;
  const T h = dot(v,base.n);
  TV dr = v-h*base.n;
  const T r = magnitude(dr);
  dr = r?dr/r:base.n.unit_orthogonal_vector();
  const T rp = r-radius,
          hp = max(-h,h-height);
  return hp>0 && rp>0 ? base.x0+clamp(h,(T)0,height)*base.n+radius*dr // outside
       : hp>rp ? X+((2*h<=height?0:height)-h)*base.n // close to end caps 
       : X+(radius-r)*dr; // close to infinite cylinder
}

T Cylinder::phi(const TV& X) const {
  const TV v = X - base.x0;
  const T h = dot(v,base.n);
  const T rp = magnitude(v-h*base.n)-radius,
          hp = max(-h,h-height);
  return hp>0 && rp>0 ? magnitude(vec(hp,rp)) : max(hp,rp);
}

T Cylinder::volume() const {
  return pi*sqr(radius)*height;
}

Box<TV> Cylinder::bounding_box() const {
  const auto n = base.n;
  const TV half = radius*TV(magnitude(vec(n.y,n.z)),magnitude(vec(n.z,n.x)),magnitude(vec(n.x,n.y)));
  const Box<TV> disk(-half,half);
  return Box<TV>::combine(base.x0+disk,base.x0+height*n+disk);
}

Vector<T,2> Cylinder::principal_curvatures(const TV& X) const {
  OTHER_NOT_IMPLEMENTED();
}

bool Cylinder::lazy_intersects(const Box<TV>& box) const {
  OTHER_NOT_IMPLEMENTED();
}

string Cylinder::repr() const {
  return format("Cylinder(%s,%s,%s)",tuple_repr(base.x0),tuple_repr(base.x0+height*base.n),other::repr(radius));
}

ostream& operator<<(ostream& output, const Cylinder& cylinder) {
  return output<<cylinder.repr();
}

PyObject* to_python(const Cylinder& cylinder) {
  OTHER_NOT_IMPLEMENTED();
}

Cylinder FromPython<Cylinder>::convert(PyObject* object) {
  OTHER_NOT_IMPLEMENTED();
}

}
