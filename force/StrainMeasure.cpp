//#####################################################################
// Class StrainMeasure
//#####################################################################
#include <other/core/force/StrainMeasure.h>
#include <other/core/array/sort.h>
#include <other/core/array/view.h>
#include <other/core/python/Class.h>
#include <other/core/vector/Matrix2x2.h>
#include <other/core/vector/Matrix3x2.h>
#include <other/core/vector/Matrix3x3.h>
#include <other/core/vector/UpperTriangularMatrix2x2.h>
#include <other/core/vector/UpperTriangularMatrix3x3.h>
#include <other/core/utility/Log.h>
namespace other{

using Log::cout;
using std::endl;

typedef real T;
template<> OTHER_DEFINE_TYPE(StrainMeasure<Vector<T,2>,2>)
template<> OTHER_DEFINE_TYPE(StrainMeasure<Vector<T,3>,2>)
template<> OTHER_DEFINE_TYPE(StrainMeasure<Vector<T,3>,3>)

template<class TV,int d> StrainMeasure<TV,d>::
StrainMeasure(Array<const Vector<int,d+1> > elements,RawArray<const TV> X)
  : nodes(elements.size()?scalar_view(elements).max()+1:0)
  , elements(elements)
  , Dm_inverse(elements.size(),false)
{
  for (int t=0;t<elements.size();t++) {
    UpperTriangularMatrix<T,d> R = Ds(X,t).R_from_QR_factorization();
    if(R.determinant()<=0) OTHER_FATAL_ERROR("Inverted or degenerate rest state");
    Dm_inverse(t)=R.inverse();}
}

template<class TV,int d> StrainMeasure<TV,d>::~StrainMeasure() {}

namespace{
template<class T> UpperTriangularMatrix<T,2> equilateral_dm(const Vector<T,2>&) {
  return UpperTriangularMatrix<T,2>(1,.5,T(sqrt(3)/2));
}
template<class T> UpperTriangularMatrix<T,3> equilateral_dm(const Vector<T,3>&) {
  T x=(T)sqrt(3)/3,d=(T).5*x,h=(T)sqrt(6)/3;
  return Matrix<T,3>(x,0,-h,-d,.5,-h,-d,-.5,-h).R_from_QR_factorization();
}}
template<class TV,int d> void StrainMeasure<TV,d>::initialize_rest_state_to_equilateral(const T side_length) {
  OTHER_ASSERT(Dm_inverse.size()==elements.size());
  UpperTriangularMatrix<T,d> Dm=side_length*equilateral_dm(Vector<T,d>());
  Dm_inverse.fill(Dm.inverse());
}

template<class TV,int d> void StrainMeasure<TV,d>::print_altitude_statistics() {   
  if (!Dm_inverse.size()) return;
  Array<T> altitude(Dm_inverse.size(),false);
  for (int t=0;t<altitude.size();t++)
    altitude(t) = rest_altitude(t);
  sort(altitude);
  Log::cout<<"strain measure - total elements = "<<altitude.size()<<std::endl;
  Log::cout<<"strain measure - smallest altitude = "<<altitude[0]<<std::endl;
  Log::cout<<"strain measure - one percent altitude = "<<altitude[(int)(.01*altitude.size())]<<std::endl;
  Log::cout<<"strain measure - ten percent altitude = "<<altitude[(int)(.1*altitude.size())]<<std::endl;
  Log::cout<<"strain measure - median altitude = "<<altitude[(int)(.5*altitude.size())]<<std::endl;
}

#define INSTANTIATION_HELPER(m,d) \
  template StrainMeasure<Vector<real,m>,d>::StrainMeasure(Array<const Vector<int,d+1> >,RawArray<const Vector<real,m> >); \
  template void StrainMeasure<Vector<real,m>,d>::initialize_rest_state_to_equilateral(const real); \
  template void StrainMeasure<Vector<real,m>,d>::print_altitude_statistics();
INSTANTIATION_HELPER(2,2)
INSTANTIATION_HELPER(3,2)
INSTANTIATION_HELPER(3,3)
}
using namespace other;

void wrap_strain_measure() {
  {typedef StrainMeasure<Vector<T,2>,2> Self;
  Class<Self>("StrainMeasure2d")
    .OTHER_INIT(Array<const Vector<int,3> >,RawArray<const Vector<T,2> >)
    .OTHER_FIELD(elements)
    .OTHER_METHOD(print_altitude_statistics)
    ;}

  {typedef StrainMeasure<Vector<T,3>,2> Self;
  Class<Self>("StrainMeasureS3d")
    .OTHER_INIT(Array<const Vector<int,3> >,RawArray<const Vector<T,3> >)
    .OTHER_FIELD(elements)
    .OTHER_METHOD(print_altitude_statistics)
    ;}

  {typedef StrainMeasure<Vector<T,3>,3> Self;
  Class<Self>("StrainMeasure3d")
    .OTHER_INIT(Array<const Vector<int,4> >,RawArray<const Vector<T,3> >)
    .OTHER_FIELD(elements)
    .OTHER_METHOD(print_altitude_statistics)
    ;}
}
