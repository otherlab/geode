//#####################################################################
// Class FrameImplicit
//#####################################################################
#include <geode/geometry/FrameImplicit.h>
#include <geode/array/Array.h>
#include <geode/array/Array2d.h>
#include <geode/array/Array3d.h>
#include <geode/python/Class.h>
namespace geode {

typedef real T;
template<> GEODE_DEFINE_TYPE(FrameImplicit<Vector<T,2> >)
template<> GEODE_DEFINE_TYPE(FrameImplicit<Vector<T,3> >)

template<class TV> FrameImplicit<TV>::
FrameImplicit(Frame<TV> frame, const Implicit<TV>& object)
  : frame(frame), object(ref(object)) {}

template<class TV> FrameImplicit<TV>::
~FrameImplicit() {}

template<class TV> typename TV::Scalar FrameImplicit<TV>::phi(const TV& X) const {
  return object->phi(frame.inverse_times(X));
}

template<class TV> TV FrameImplicit<TV>::normal(const TV& X) const {
  return frame.r*object->normal(frame.inverse_times(X));
}

template<class TV> TV FrameImplicit<TV>::surface(const TV& X) const {
  return frame*object->surface(frame.inverse_times(X));
}

template<class TV> bool FrameImplicit<TV>::lazy_inside(const TV& X) const {
  return object->lazy_inside(frame.inverse_times(X));
}

template<class TV> Box<TV> FrameImplicit<TV>::bounding_box() const {
  Array<TV,Base::d> corners;
  object->bounding_box().corners(corners);
  return geode::bounding_box(frame*corners.flat);
}

template<class TV> string FrameImplicit<TV>::repr() const {
  GEODE_NOT_IMPLEMENTED();
  //return format("FrameImplicit(%s,%s)",geode::repr(frame),object->repr());
}

template class FrameImplicit<Vector<T,2> >;
template class FrameImplicit<Vector<T,3> >;

}
using namespace geode;

template<int d> static void wrap_helper() {
  typedef Vector<T,d> TV;
  typedef FrameImplicit<TV> Self;
  static const string name = format("FrameImplicit%dd",d);

  Class<Self>(name.c_str())
    .GEODE_INIT(Frame<TV>,const Implicit<TV>&)
    ;
}

void wrap_frame_implicit() {
  wrap_helper<2>();
  wrap_helper<3>();
}
