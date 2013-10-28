//#####################################################################
// Header Register
//#####################################################################
#pragma once

#include <geode/array/forward.h>
#include <geode/utility/config.h>
#include <geode/vector/forward.h>
#include <geode/array/RawArray.h>
namespace geode {

GEODE_CORE_EXPORT Frame<Vector<real,2>> rigid_register(RawArray<const Vector<real,2>> X0, RawArray<const Vector<real,2>> X1, RawArray<const real> mass = RawArray<const real>());
GEODE_CORE_EXPORT Frame<Vector<real,3>> rigid_register(RawArray<const Vector<real,3>> X0, RawArray<const Vector<real,3>> X1, RawArray<const real> mass = RawArray<const real>());

// Find the best affine transform from X0 to X1.  Warning: undetermined problems will give poor results.
GEODE_CORE_EXPORT Matrix<real,3> affine_register(RawArray<const Vector<real,2>> X0, RawArray<const Vector<real,2>> X1, RawArray<const real> mass = RawArray<const real>());
GEODE_CORE_EXPORT Matrix<real,4> affine_register(RawArray<const Vector<real,3>> X0, RawArray<const Vector<real,3>> X1, RawArray<const real> mass = RawArray<const real>());

}
