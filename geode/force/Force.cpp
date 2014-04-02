//#####################################################################
// Class Force
//#####################################################################
#include <geode/force/Force.h>
#include <geode/vector/SolidMatrix.h>
#include <geode/vector/SymmetricMatrix.h>
namespace geode {

typedef real T;
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
