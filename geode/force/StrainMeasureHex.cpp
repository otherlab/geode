//#####################################################################
// Class StrainMeasureHex
//#####################################################################
#include <geode/force/StrainMeasureHex.h>
#include <geode/array/view.h>
#include <geode/python/Class.h>
#include <geode/utility/Log.h>
#include <geode/vector/Matrix.h>
namespace geode {

using Log::cout;
using std::endl;

typedef real T;
typedef Vector<T,3> TV;
GEODE_DEFINE_TYPE(StrainMeasureHex)

static Vector<Matrix<T,8,3>,8> initialize_Hg() {
  Vector<Matrix<T,8,3>,8> H;
  for(int i=0;i<2;i++) for(int j=0;j<2;j++) for(int k=0;k<2;k++) {
    int g = 4*i+2*j+k;
    TV xi=(T)sqrt(1./3)*TV((T)(2*i-1),(T)(2*j-1),(T)(2*k-1));
    for(int ii=0;ii<2;ii++) for(int jj=0;jj<2;jj++) for(int kk=0;kk<2;kk++) {
      TV s(2*ii-1,2*jj-1,2*kk-1),sxi=s*xi;
      H[g][4*ii+2*jj+kk] = (T).125*s*TV((1+sxi.y)*(1+sxi.z),(1+sxi.x)*(1+sxi.z),(1+sxi.x)*(1+sxi.y));
    }
  }
  return H;
}

const Vector<Matrix<T,8,3>,8> StrainMeasureHex::Hg = initialize_Hg();

StrainMeasureHex::StrainMeasureHex(Array<const Vector<int,8>> elements, Array<const TV> X)
  : nodes(elements.size()?scalar_view(elements).max()+1:0)
  , elements(elements)
  , rest_X(X)
  , H_DmH_inverse(elements.size(),uninit)
  , DmH_det(elements.size(),uninit) {
  GEODE_ASSERT(nodes<=X.size());
  for (int h=0;h<elements.size();h++)
    for (int g=0;g<8;g++) {
      Matrix<T,3> DmH;
      for (int k=0;k<8;k++)
        DmH += outer_product(X[elements[h][k]],Hg[g][k]);
      DmH_det[h][g] = DmH.determinant();
      GEODE_ASSERT(DmH_det[h][g]>0);
      H_DmH_inverse[h][g] = Hg[g]*DmH.inverse();
    }
}

StrainMeasureHex::~StrainMeasureHex() {}

T StrainMeasureHex::DmH_minimum_altitude(int hex) const {
  T result = FLT_MAX;
  for (int g=0;g<8;g++) {
    Matrix<T,3> DmH;
    for (int k=0;k<8;k++)
      DmH += outer_product(rest_X[elements[hex][k]],Hg[g][k]);
    result = min(result,DmH.simplex_minimum_altitude());
  }
  return result;
}

Matrix<T,3> StrainMeasureHex::gradient(RawArray<const TV> X,int hex,TV w) const {
  TV xi = (T)2*w-1;
  Matrix<T,8,3> H;
  for (int ii=0;ii<2;ii++) for (int jj=0;jj<2;jj++) for (int kk=0;kk<2;kk++) {
    TV s(2*ii-1,2*jj-1,2*kk-1),sxi=s*xi;
    H[4*ii+2*jj+kk] = (T).125*s*TV((1+sxi.y)*(1+sxi.z),(1+sxi.x)*(1+sxi.z),(1+sxi.x)*(1+sxi.y));
  }
  Matrix<T,3> DmH;
  for (int k=0;k<8;k++)
    DmH += outer_product(rest_X[elements[hex][k]],H[k]);
  Matrix<T,3> DmH_inv = DmH.inverse();
  Matrix<T,3> gradient;
  for (int k=0;k<8;k++)
    gradient += outer_product(X[elements[hex][k]],DmH_inv.transpose_times(H[k]));
  return gradient;
}

}
using namespace geode;

void wrap_strain_measure_hex() {
  typedef StrainMeasureHex Self;
  Class<Self>("StrainMeasureHex")
    .GEODE_INIT(Array<const Vector<int,8>>,Array<const Vector<T,3>>)
    .GEODE_FIELD(elements)
    ;
}
