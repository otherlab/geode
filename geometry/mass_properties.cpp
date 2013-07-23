//#####################################################################
// Mass properties of curves and surfaces
//#####################################################################
#include <other/core/geometry/mass_properties.h>
#include <other/core/vector/Frame.h>
#include <other/core/vector/Matrix.h>
#include <other/core/vector/DiagonalMatrix.h>
#include <other/core/vector/SymmetricMatrix.h>
#include <other/core/force/StrainMeasure.h>
#include <other/core/math/Factorial.h>
namespace other{

namespace{
template<class T,int d> inline T scaled_element_covariance(const T scaled_element_volume,const Matrix<T,2,d>& DX) // actually returns only the trace of the covariance matrix
{
    static const SymmetricMatrix<T,d> canonical=(T)1+SymmetricMatrix<T,d>::unit_matrix();
    return scaled_element_volume*inner_product(DX*canonical,DX);
}
template<class T,int d> inline SymmetricMatrix<T,3> scaled_element_covariance(const T scaled_element_volume,const Matrix<T,3,d>& DX)
{
    static const SymmetricMatrix<T,d> canonical=(T)1+SymmetricMatrix<T,d>::unit_matrix();
    return conjugate(DX,scaled_element_volume*canonical);
}
template<class T> inline T inertia_tensor_from_covariance(const T covariance_trace)
{
    return covariance_trace;
}
template<class T> inline SymmetricMatrix<T,3> inertia_tensor_from_covariance(const SymmetricMatrix<T,3>& covariance)
{
    return covariance.trace()-covariance;
}}

template<bool filled,class TV,int s> static MassProperties<TV>
helper(RawArray<const Vector<int,s> > elements, RawArray<const TV> X) {
  typedef typename TV::Scalar T;
  static const int d = s-1;
  MassProperties<TV> props;

  // Compute center and volume
  const TV base = X[elements(0)[0]];
  T scaled_volume = 0; // (d+filled)!*volume
  TV scaled_center_times_volume; // (d+1+filled)!*center*volume
  for(int t=0;t<elements.size();t++){
    const Vector<int,d+1>& nodes = elements[t];
    Matrix<T,TV::m,d+1> DX;
    for(int i=0;i<nodes.m;i++) DX.set_column(i,X[nodes[i]]-base);
    T scaled_element_volume = filled?DX.parallelepiped_measure():StrainMeasure<T,d>::Ds(X,nodes).parallelepiped_measure();
    scaled_volume += scaled_element_volume;
    scaled_center_times_volume += scaled_element_volume*DX.column_sum();}
  props.volume = (T)1/Factorial<d+filled>::value*scaled_volume;
  if (!props.volume)
    OTHER_FATAL_ERROR("zero volume");
  props.center = base+(T)1/(d+1+filled)/scaled_volume*scaled_center_times_volume;

  // Compute inertia tensor: see http://number-none.com/blow/inertia for explanation of filled case
  // During loop, props.inertia_tensor contains (d+2+filled)!*covariance (or trace(covariance) in 2d)
  for(int t=0;t<elements.size();t++){
      const Vector<int,d+1>& nodes = elements[t];
      Matrix<T,TV::m,d+1> DX;
      for(int i=0;i<nodes.m;i++) DX.set_column(i,X[nodes[i]]-props.center);
      T scaled_element_volume = filled?DX.parallelepiped_measure():StrainMeasure<T,d>::Ds(X,nodes).parallelepiped_measure();
      props.inertia_tensor += scaled_element_covariance(scaled_element_volume,DX);}
  props.inertia_tensor = inertia_tensor_from_covariance((T)1/Factorial<d+2+filled>::value*props.inertia_tensor);
  return props;
}

template<class TV,int s> MassProperties<TV>
mass_properties(RawArray<const Vector<int,s> > elements, RawArray<const TV> X, bool filled) {
  if (!elements.size())
    return MassProperties<TV>();
  if (filled && s!=TV::m)
    OTHER_FATAL_ERROR("only codimension 1 objects can be filled");
  return filled?helper<true>(elements,X):helper<false>(elements,X);
}

template<class TV,int s> Frame<TV> principal_frame(RawArray<const Vector<int,s> > elements, RawArray<const TV> X, bool filled) {
  typedef typename TV::Scalar T;
  MassProperties<TV> props = mass_properties(elements,X,filled);
  Matrix<T,TV::m> U;DiagonalMatrix<T,TV::m> D;
  props.inertia_tensor.fast_solve_eigenproblem(D,U);
  return Frame<TV>(props.center,Rotation<TV>(U));
}

typedef real T;
#define INSTANTIATE(m,d) \
  template MassProperties<Vector<T,m> > mass_properties(RawArray<const Vector<int,d+1> > elements, RawArray<const Vector<T,m> > X, bool filled);
template Frame<Vector<T,3> > principal_frame(RawArray<const Vector<int,3> > elements, RawArray<const Vector<T,3> > X, bool filled);
INSTANTIATE(2,1)
INSTANTIATE(3,1)
INSTANTIATE(3,2)

}
