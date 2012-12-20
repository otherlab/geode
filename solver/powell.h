#pragma once

// Powell's method for multidimensional optimization.  The implementation
// is taken from scipy, which requires the following notice:
// ******NOTICE***************
// optimize.py module by Travis E. Oliphant
//
// You may copy and use this module as you see fit with no
// guarantee implied provided you keep this notice in all copies.
// *****END NOTICE************

#include <other/core/array/RawArray.h>
#include <other/core/structure/Tuple.h>
#include <boost/function.hpp>
namespace other {

using boost::function;

// Minimize a multidimensional function using Powell's method.  Unlike scipy, all tolerances are absolute.
// Arguments:
//   x: starting point
//   scale: initial step size
//   xtol: absolute point tolerance
//   ftol: absolute function value tolerance
OTHER_CORE_EXPORT Tuple<real,int> powell(const function<real(RawArray<const real>)>& f, RawArray<real> x, real scale, real xtol, real ftol, int maxiter);

}
