//#####################################################################
// Class StrainMeasureHex
//#####################################################################
#pragma once

#include <other/core/array/Array.h>
#include <other/core/vector/Matrix.h>
#include <other/core/vector/SymmetricMatrix.h>
namespace other{

class StrainMeasureHex : public Object {
  typedef real T;
  typedef Vector<T,3> TV;
public:
  OTHER_DECLARE_TYPE
  typedef Object Base;

  const int nodes;
  const Array<const Vector<int,8> > elements;
  const Array<const TV> rest_X;
  Array<Vector<Matrix<T,8,3>,8> > H_DmH_inverse; // 8x3 matrix per gauss point per element
  Array<Vector<T,8> > DmH_det;
private:
  static const Vector<Matrix<T,8,3>,8> Hg;

protected:
  StrainMeasureHex(Array<const Vector<int,8> > elements,Array<const TV> X);
public:
  ~StrainMeasureHex();

  Matrix<T,3> gradient(RawArray<const TV> X,int hex,int gauss) const {
    Matrix<T,3> gradient;
    for(int k=0;k<8;k++)
      gradient += outer_product(X[elements[hex][k]],H_DmH_inverse[hex][gauss][k]);
    return gradient;
  }

  // scaled_stress should be effective_volume*stress
  void distribute_stress(RawArray<TV> F,const SymmetricMatrix<T,3>& scaled_stress,int hex,int gauss) const {
    for(int k=0;k<8;k++) {
      TV h = H_DmH_inverse[hex][gauss][k];
      F[elements[hex][k]] -= scaled_stress*h;
    }
  }

  OTHER_CORE_EXPORT Matrix<T,3> gradient(RawArray<const TV> X,int hex,TV w) const; // w in [0,1]^3
  T DmH_minimum_altitude(int hex) const; // this is only vaguely an altitude
};

}
