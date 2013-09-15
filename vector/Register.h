//#####################################################################
// Header Register
//#####################################################################
#pragma once

#include <other/core/array/forward.h>
#include <other/core/utility/config.h>
#include <other/core/vector/forward.h>
namespace other {

OTHER_CORE_EXPORT Frame<Vector<real,2>> rigid_register(RawArray<const Vector<real,2>> X0, RawArray<const Vector<real,2>> X1, RawArray<const real> mass = RawArray<const real>());
OTHER_CORE_EXPORT Frame<Vector<real,3>> rigid_register(RawArray<const Vector<real,3>> X0, RawArray<const Vector<real,3>> X1, RawArray<const real> mass = RawArray<const real>());

// Find the best affine transform from X0 to X1.  Warning: undetermined problems will give poor results.
OTHER_CORE_EXPORT Matrix<real,3> affine_register(RawArray<const Vector<real,2>> X0, RawArray<const Vector<real,2>> X1, RawArray<const real> mass = RawArray<const real>());
OTHER_CORE_EXPORT Matrix<real,4> affine_register(RawArray<const Vector<real,3>> X0, RawArray<const Vector<real,3>> X1, RawArray<const real> mass = RawArray<const real>());

}
