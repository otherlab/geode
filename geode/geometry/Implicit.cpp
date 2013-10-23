//#####################################################################
// Class Implicit
//#####################################################################
#include <geode/geometry/Implicit.h>
#include <geode/python/Class.h>
namespace geode {

typedef real T;
template<> GEODE_DEFINE_TYPE(Implicit<Vector<T,1> >)
template<> GEODE_DEFINE_TYPE(Implicit<Vector<T,2> >)
template<> GEODE_DEFINE_TYPE(Implicit<Vector<T,3> >)

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
using namespace geode;

template<int d> static void wrap_helper() {
  typedef Vector<T,d> TV;
  typedef Implicit<TV> Self;

  Class<Self>("Implicit")
    .GEODE_FIELD(d)
    .GEODE_METHOD(phi)
    .GEODE_METHOD(normal)
    .GEODE_METHOD(lazy_inside)
    .GEODE_METHOD(surface)
    .GEODE_METHOD(bounding_box)
    .GEODE_REPR()
    ;
}

void wrap_implicit() {
  wrap_helper<1>();
  wrap_helper<2>();
  wrap_helper<3>();
}
