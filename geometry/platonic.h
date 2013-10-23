// Convenience functions for generating platonic solids
#pragma once

#include <other/core/mesh/TriangleSoup.h>
#include <other/core/structure/Tuple.h>
namespace other {

OTHER_CORE_EXPORT Tuple<Ref<TriangleSoup>,Array<Vector<real,3>>> icosahedron_mesh();
OTHER_CORE_EXPORT Tuple<Ref<TriangleSoup>,Array<Vector<real,3>>> sphere_mesh(const int refinements, const Vector<real,3> center=(Vector<real,3>()), const real radius=1);

OTHER_CORE_EXPORT Ref<TriangleSoup> double_torus_mesh();

}
