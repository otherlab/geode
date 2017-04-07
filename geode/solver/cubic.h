#pragma once
#include <geode/array/SmallArray.h>
#include <geode/vector/Vector.h>
namespace geode {

// Find the one or three real roots of the cubic
SmallArray<real,3> solve_cubic(const real a, const real b, const real c, const real d);

} // geode namespace