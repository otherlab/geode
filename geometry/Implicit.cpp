//#####################################################################
// Class Implicit
//#####################################################################
#include <other/core/geometry/Implicit.h>
#include <other/core/python/Class.h>
namespace other{

typedef real T;
template<> OTHER_DEFINE_TYPE(Implicit<Vector<T,1> >)
template<> OTHER_DEFINE_TYPE(Implicit<Vector<T,2> >)
template<> OTHER_DEFINE_TYPE(Implicit<Vector<T,3> >)

template<class TV> const int Implicit<TV>::d;

template<class TV> Implicit<TV>::
Implicit()
{}

template<class TV> Implicit<TV>::
~Implicit()
{}

template class Implicit<Vector<T,1> >;
template class Implicit<Vector<T,2> >;
template class Implicit<Vector<T,3> >;

}
using namespace other;

template<int d> static void wrap_helper() {
  typedef Vector<T,d> TV;
  typedef Implicit<TV> Self;

  Class<Self>("Implicit")
    .OTHER_FIELD(d)
    .OTHER_METHOD(phi)
    .OTHER_METHOD(normal)
    .OTHER_METHOD(lazy_inside)
    .OTHER_METHOD(surface)
    .OTHER_METHOD(bounding_box)
    .OTHER_REPR()
    ;
}

void wrap_implicit() {
  wrap_helper<1>();
  wrap_helper<2>();
  wrap_helper<3>();
}
