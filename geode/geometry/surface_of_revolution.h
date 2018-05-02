#pragma once
#include <geode/array/Array.h>
#include <geode/vector/Vector.h>
#include <geode/structure/Tuple.h>
namespace geode {

// Generates mesh by rotating profile_rz around z axis
// Returns triangles and positions of vertices

GEODE_CORE_EXPORT Tuple<Array<Vector<int,3>>,Array<Vector<float,3>>>
    surface_of_revolution(const RawArray<const Vector<float,2>> profile_rz, const int sides);

GEODE_CORE_EXPORT Tuple<Array<Vector<int,3>>,Array<Vector<double,3>>>
    surface_of_revolution(const RawArray<const Vector<double,2>> profile_rz, const int sides);

} // geode namespace
