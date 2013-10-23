// Fast ray-box intersection tests
#pragma once

#include <geode/geometry/Box.h>
#include <geode/geometry/Ray.h>
namespace geode {
namespace {

// We use the ray-box intersection algorithm from
//   Williams, Barrus, Morley, and Shirley, "An efficient and robust ray-box intersection algorithm", http://www.cs.utah.edu/~rmorley/pubs/box.pdf.
// For speed, we templatize the code over the octant of the ray.

template<class TV,int signs> struct FastRay {
  typedef typename TV::Scalar T;

  const TV start;
  const TV inv_dx;
  T t_max;

  FastRay(const Ray<TV>& ray)
    : start(ray.start), inv_dx(1/ray.direction), t_max(ray.t_max) {
    for (int i=0;i<TV::m;i++)
      assert(sign(i)==(inv_dx[i]<0));
  }

  bool sign(const int i) const {
    return (signs&1<<i)!=0;
  }

  Box<T> range(const Box<Vector<T,2>>& box, const T box_enlargement) const {
    const TV bounds[2] = {box.min-box_enlargement,box.max+box_enlargement};
    double lo =  inv_dx.x * (bounds[  sign(0)].x - start.x),
           hi =  inv_dx.x * (bounds[1-sign(0)].x - start.x);
    lo = max(lo, inv_dx.y * (bounds[  sign(1)].y - start.y));
    hi = min(hi, inv_dx.y * (bounds[1-sign(1)].y - start.y));
    return Box<T>(lo,hi);
  }

  Box<T> range(const Box<Vector<T,3>>& box, const T box_enlargement) const {
    const TV bounds[2] = {box.min-box_enlargement,box.max+box_enlargement};
    double lo =  inv_dx.x * (bounds[  sign(0)].x - start.x),
           hi =  inv_dx.x * (bounds[1-sign(0)].x - start.x);
    lo = max(lo, inv_dx.y * (bounds[  sign(1)].y - start.y));
    hi = min(hi, inv_dx.y * (bounds[1-sign(1)].y - start.y));
    lo = max(lo, inv_dx.z * (bounds[  sign(2)].z - start.z));
    hi = min(hi, inv_dx.z * (bounds[1-sign(2)].z - start.z));
    return Box<T>(lo,hi);
  }

  bool intersects(const Box<TV>& box, const T box_enlargement) const {
    const auto ts = range(box,box_enlargement);
    return ts.min<=ts.max && 0<=ts.max && ts.min<=t_max;
  }
};

template<class TV> static inline int fast_ray_signs(const Ray<TV>& ray) {
  const TV inv_dx = 1/ray.direction; 
  int signs = 0;
  for (int i=0;i<TV::m;i++)
    signs |= (inv_dx[i]<0)<<i;
  return signs;
}

}
}
