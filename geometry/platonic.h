// Convenience functions for generating platonic solids
#pragma once

#include <other/core/mesh/TriangleMesh.h>
#include <other/core/structure/Tuple.h>
namespace other {

Tuple<Ref<TriangleMesh>,Array<Vector<real,3>>> icosahedron_mesh() OTHER_EXPORT;
Tuple<Ref<TriangleMesh>,Array<Vector<real,3>>> sphere_mesh(const int refinements, const Vector<real,3> center=(Vector<real,3>()), const real radius=1) OTHER_EXPORT;

}