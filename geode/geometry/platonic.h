// Convenience functions for generating platonic solids
#pragma once

#include <geode/mesh/TriangleSoup.h>
#include <geode/structure/Tuple.h>

namespace geode {

GEODE_CORE_EXPORT Tuple<Ref<TriangleSoup>,Array<Vector<real,3>>> icosahedron_mesh();
GEODE_CORE_EXPORT Tuple<Ref<TriangleSoup>,Array<Vector<real,3>>> sphere_mesh(const int refinements, const Vector<real,3> center=(Vector<real,3>()), const real radius=1);

GEODE_CORE_EXPORT Ref<TriangleSoup> double_torus_mesh();

GEODE_CORE_EXPORT Tuple<Ref<TriangleSoup>, Array<Vector<real,3>>> cube_mesh(Vector<real,3> const &min, Vector<real,3> const &max);

}
