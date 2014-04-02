//#####################################################################
// Class Box
//#####################################################################
#include <geode/geometry/Box.h>
#include <geode/geometry/AnalyticImplicit.h>
#include <geode/geometry/Ray.h>
#include <geode/array/NdArray.h>
#include <geode/array/Nested.h>
#include <geode/array/view.h>
#include <geode/utility/exceptions.h>
#include <geode/utility/Ptr.h>
#include <geode/math/cube.h>
#include <geode/utility/format.h>
#include <geode/structure/Tuple.h>
namespace geode {

template<class T,int d> Vector<T,d> Box<Vector<T,d>>::surface(const TV& X) const {
  if (!lazy_inside(X)) return geode::clamp(X,min,max);
  TV min_side = X-min,
     max_side = max-X,
     result = X;
  int min_argmin = min_side.argmin(),
      max_argmin = max_side.argmin();
  if (min_side[min_argmin]<=max_side[max_argmin])
    result[min_argmin] = min[min_argmin];
  else
    result[max_argmin] = max[max_argmin];
  return result;
}

template<class T,int d> T Box<Vector<T,d>>::phi(const TV& X) const {
  TV lengths = sizes();
  TV phi = abs(X-center())-(T).5*lengths;
  if (!all_less_equal(phi,TV()))
    return TV::componentwise_max(phi,TV()).magnitude();
  return phi.max();
}

template<class T,int d> Vector<T,d> Box<Vector<T,d>>::normal(const TV& X) const {
  if (lazy_inside(X)) {
    TV phis_max = X-max,
       phis_min = min-X;
    int axis = TV::componentwise_max(phis_min,phis_max).argmax();
    return T(phis_max[axis]>phis_min[axis]?1:-1)*TV::axis_vector(axis);
  } else {
    TV phis_max = X-max,
       phis_min = min-X;
    TV normal;
    for (int i=0;i<d;i++) {
      T phi = geode::max(phis_min[i],phis_max[i]);
      normal[i] = phi>0?(phis_max[i]>phis_min[i]?phi:-phi):0;
    }
    return normal.normalized();
  }
}

template<class T,int d> string Box<Vector<T,d>>::name() {
  return format("Box<Vector<T,%d>",d);
}

template<class T,int d> string Box<Vector<T,d>>::repr() const {
  return format("Box(%s,%s)",tuple_repr(min),tuple_repr(max));
}

#define INSTANTIATION_HELPER(T,d) \
  template GEODE_CORE_EXPORT string Box<Vector<T,d>>::name(); \
  template GEODE_CORE_EXPORT string Box<Vector<T,d>>::repr() const; \
  template GEODE_CORE_EXPORT Vector<T,d> Box<Vector<T,d>>::normal(const Vector<T,d>&) const; \
  template GEODE_CORE_EXPORT Vector<T,d> Box<Vector<T,d>>::surface(const Vector<T,d>&) const; \
  template GEODE_CORE_EXPORT Vector<T,d>::Scalar Box<Vector<T,d>>::phi(const Vector<T,d>&) const;
INSTANTIATION_HELPER(real,1)
INSTANTIATION_HELPER(real,2)
INSTANTIATION_HELPER(real,3)

}
