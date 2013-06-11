#pragma once

#include <other/core/vector/Vector.h>
#include <boost/function.hpp>
namespace other {

using boost::function;

// Maximize a functional over a sphere using pattern search
OTHER_CORE_EXPORT real spherical_pattern_maximize(const function<real(Vector<real,3>)>& score, Vector<real,3>& n, real tol);

}
