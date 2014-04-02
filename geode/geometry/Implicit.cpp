//#####################################################################
// Class Implicit
//#####################################################################
#include <geode/geometry/Implicit.h>
namespace geode {

typedef real T;

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
