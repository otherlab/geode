#pragma once

#include <geode/vector/Vector.h>
#include <boost/function.hpp>
namespace geode {

using boost::function;

// Maximize a functional over a sphere using pattern search
GEODE_CORE_EXPORT real spherical_pattern_maximize(const function<real(Vector<real,3>)>& score, Vector<real,3>& n, real tol);

}
