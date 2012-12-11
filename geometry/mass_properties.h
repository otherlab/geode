//#####################################################################
// Mass properties of curves and surfaces
//#####################################################################
#pragma once

#include <other/core/array/forward.h>
#include <other/core/utility/config.h>
#include <other/core/vector/forward.h>
namespace other{

template<class TV> struct InertiaTensorPolicy;
template<class T> struct InertiaTensorPolicy<Vector<T,1> >{typedef Matrix<T,0> WorldSpace;};
template<class T> struct InertiaTensorPolicy<Vector<T,2> >{typedef T WorldSpace;};
template<class T> struct InertiaTensorPolicy<Vector<T,3> >{typedef SymmetricMatrix<T,3> WorldSpace;};

template<class TV> struct MassProperties {
  typename TV::Scalar volume;
  TV center;
  typename InertiaTensorPolicy<TV>::WorldSpace inertia_tensor;

  MassProperties()
    :volume(),center(),inertia_tensor() {}
};

template<class TV,int s> MassProperties<TV> mass_properties(RawArray<const Vector<int,s> > elements, RawArray<const TV> X, bool filled) OTHER_CORE_EXPORT;
template<class TV,int s> Frame<TV> principal_frame(RawArray<const Vector<int,s> > elements, RawArray<const TV> X, bool filled) OTHER_CORE_EXPORT;

}
