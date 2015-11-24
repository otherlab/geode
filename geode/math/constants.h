//#####################################################################
// Header constants
//#####################################################################
#pragma once
#include <geode/config.h>

#include <cmath>
#include <limits>
namespace geode {

using std::numeric_limits;

#ifdef _WIN32
GEODE_CONSTEXPR_IF_NOT_MSVC double pi = 3.14159265358979323846;
#else
GEODE_CONSTEXPR_IF_NOT_MSVC double pi = M_PI;
#endif
GEODE_CONSTEXPR_IF_NOT_MSVC double half_pi = 0.5*pi;
GEODE_CONSTEXPR_IF_NOT_MSVC double tau = 2*pi;
GEODE_CONSTEXPR_IF_NOT_MSVC double speed_of_light = 2.99792458e8; // m/s
GEODE_CONSTEXPR_IF_NOT_MSVC double plancks_constant = 6.6260755e-34; // J*s
GEODE_CONSTEXPR_IF_NOT_MSVC double boltzmanns_constant = 1.380658e-23; // J/K
GEODE_CONSTEXPR_IF_NOT_MSVC double ideal_gas_constant = 8.314472; // J/K/mol

const double inf = numeric_limits<double>::infinity();

template<int d> struct unit_sphere_size{static_assert(d<4,"");static const double value;};
template<int d> const double unit_sphere_size<d>::value=d==0?0:d==1?2:d==2?pi:4*pi/3;

}
