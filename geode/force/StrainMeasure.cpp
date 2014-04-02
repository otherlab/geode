//#####################################################################
// Class StrainMeasure
//#####################################################################
#include <geode/force/StrainMeasure.h>
#include <geode/array/Array2d.h>
#include <geode/array/sort.h>
#include <geode/array/view.h>
#include <geode/math/constants.h>
#include <geode/vector/Matrix.h>
#include <geode/vector/UpperTriangularMatrix.h>
#include <geode/utility/Log.h>
namespace geode {

using Log::cout;
using std::endl;

typedef real T;

template<int d,int m> static Array<UpperTriangularMatrix<T,d>> compute_Dm_inverse(RawArray<const Vector<int,d+1>> elements, RawArray<const Vector<T,m>> X) {
  Array<UpperTriangularMatrix<T,d>> Dm_inverse(elements.size(),uninit);
  for (int t=0;t<elements.size();t++) {
    const auto R = StrainMeasure<T,d>::Ds(X,elements[t]).R_from_QR_factorization();
    if (R.determinant()<=0)
      throw RuntimeError("StrainMeasure: Inverted or degenerate rest state");
    Dm_inverse[t] = R.inverse();
  }
  return Dm_inverse;
}

template<int d> static Array<UpperTriangularMatrix<T,d>> compute_Dm_inverse(const int nodes, RawArray<const Vector<int,d+1>> elements, RawArray<const T,2> X) {
  GEODE_ASSERT(X.m >= nodes);
  switch (X.n) {
    case 2: return compute_Dm_inverse<d>(elements,vector_view<(d==2?2:3)>(X.flat));
    case 3: return compute_Dm_inverse<d>(elements,vector_view<3         >(X.flat));
    default: throw RuntimeError(format("StrainMeasure: Can't initialize %dD strain measure from %dD rest states",d,X.n));
  }
}

template<class T,int d> StrainMeasure<T,d>::StrainMeasure(Array<const Vector<int,d+1>> elements, RawArray<const T,2> X)
  : nodes(elements.size()?scalar_view(elements).max()+1:0)
  , elements(elements)
  , Dm_inverse(compute_Dm_inverse<d>(nodes,elements,X)) {}

template<class T,int d> StrainMeasure<T,d>::~StrainMeasure() {}

template<class T,int d> T StrainMeasure<T,d>::minimum_rest_altitude() const {
  T altitude = (T)inf;
  for (int t=0;t<Dm_inverse.m;t++)
    altitude = min(altitude,rest_altitude(t));
  return altitude;
}

template<class T> static UpperTriangularMatrix<T,2> equilateral_Dm(const Vector<T,2>&) {
  return UpperTriangularMatrix<T,2>(1,.5,T(sqrt(3)/2));
}

template<class T> static UpperTriangularMatrix<T,3> equilateral_Dm(const Vector<T,3>&) {
  const T x = (T)sqrt(3)/3,d=(T).5*x,
          h = (T)sqrt(6)/3;
  return Matrix<T,3>(x,0,-h,-d,.5,-h,-d,-.5,-h).R_from_QR_factorization();
}

template<class T,int d> void StrainMeasure<T,d>::initialize_rest_state_to_equilateral(const T side_length) {
  GEODE_ASSERT(Dm_inverse.size()==elements.size());
  UpperTriangularMatrix<T,d> Dm = side_length*equilateral_Dm(Vector<T,d>());
  Dm_inverse.fill(Dm.inverse());
}

template<class T,int d> void StrainMeasure<T,d>::print_altitude_statistics() {   
  if (!Dm_inverse.size())
    return;
  Array<T> altitude(Dm_inverse.size(),uninit);
  for (int t=0;t<altitude.size();t++)
    altitude(t) = rest_altitude(t);
  sort(altitude);
  Log::cout<<"strain measure - total elements = "<<altitude.size()<<std::endl;
  Log::cout<<"strain measure - smallest altitude = "<<altitude[0]<<std::endl;
  Log::cout<<"strain measure - one percent altitude = "<<altitude[(int)(.01*altitude.size())]<<std::endl;
  Log::cout<<"strain measure - ten percent altitude = "<<altitude[(int)(.1*altitude.size())]<<std::endl;
  Log::cout<<"strain measure - median altitude = "<<altitude[(int)(.5*altitude.size())]<<std::endl;
}

#define INSTANTIATION_HELPER(d) \
  template StrainMeasure<real,d>::StrainMeasure(Array<const Vector<int,d+1>>,RawArray<const real,2>); \
  template void StrainMeasure<real,d>::initialize_rest_state_to_equilateral(const real); \
  template void StrainMeasure<real,d>::print_altitude_statistics();
INSTANTIATION_HELPER(2)
INSTANTIATION_HELPER(3)

}
