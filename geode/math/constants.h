//#####################################################################
// Header constants
//#####################################################################
#pragma once

#include <cmath>
#include <limits>
namespace geode {

using std::numeric_limits;

#ifdef _WIN32
const double pi = 3.14159265358979323846;
#else
const double pi = M_PI;
#endif
const double speed_of_light = 2.99792458e8; // m/s
const double plancks_constant = 6.6260755e-34; // J*s
const double boltzmanns_constant = 1.380658e-23; // J/K
const double ideal_gas_constant = 8.314472; // J/K/mol

const double inf = numeric_limits<double>::infinity();

template<int d> struct unit_sphere_size{static_assert(d<4,"");static const double value;};
template<int d> const double unit_sphere_size<d>::value=d==0?0:d==1?2:d==2?pi:4*pi/3;

}
