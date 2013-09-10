//#####################################################################
// Header Register
//#####################################################################
#pragma once

#include <other/core/array/forward.h>
#include <other/core/utility/config.h>
#include <other/core/vector/forward.h>
namespace other {

OTHER_CORE_EXPORT Frame<Vector<real,2> > rigid_register(RawArray<const Vector<real,2> > X0,RawArray<const Vector<real,2> > X1, RawArray<const real> M = RawArray<const real>());
OTHER_CORE_EXPORT Frame<Vector<real,3> > rigid_register(RawArray<const Vector<real,3> > X0,RawArray<const Vector<real,3> > X1, RawArray<const real> M = RawArray<const real>());
}
