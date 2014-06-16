//#####################################################################
// Class Force
//#####################################################################
#include <geode/force/Force.h>
#include <geode/python/Class.h>
#include <geode/vector/SolidMatrix.h>
#include <geode/vector/SymmetricMatrix.h>
namespace geode {

typedef real T;
template<> GEODE_DEFINE_TYPE(Force<Vector<T,2>>)
template<> GEODE_DEFINE_TYPE(Force<Vector<T,3>>)
template<class TV> const int Force<TV>::d;

template<class TV> Force<TV>::Force() {}

template<class TV> Force<TV>::~Force() {}

template<class TV> Array<TV> Force<TV>::elastic_gradient_block_diagonal_times(RawArray<TV> dX) const {
  Array<SymmetricMatrix<T,d>> dFdX(dX.size());
  add_elastic_gradient_block_diagonal(dFdX);
  Array<TV> dF(dX.size(),uninit);
  for (int i=0;i<dX.size();i++)
    dF[i] = dFdX[i]*dX[i];
  return dF;
}

template class Force<Vector<T,2>>;
template class Force<Vector<T,3>>;
}
using namespace geode;

template<int d> static void wrap_helper() {
  typedef Force<Vector<T,d>> Self;
  Class<Self>(d==2?"Force2d":"Force3d")
    .GEODE_FIELD(d)
    .GEODE_METHOD(nodes)
    .GEODE_METHOD(update_position)
    .GEODE_METHOD(elastic_energy)
    .GEODE_METHOD(add_elastic_force)
    .GEODE_METHOD(add_elastic_differential)
    .GEODE_METHOD(damping_energy)
    .GEODE_METHOD(add_damping_force)
    .GEODE_METHOD(add_frequency_squared)
    .GEODE_METHOD(strain_rate)
    .GEODE_METHOD(structure)
    .GEODE_METHOD(add_elastic_gradient)
    .GEODE_METHOD(add_damping_gradient)
    .GEODE_METHOD(elastic_gradient_block_diagonal_times)
    ;
}

void wrap_Force() {
  wrap_helper<2>();
  wrap_helper<3>();
}
