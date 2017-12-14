#pragma once
#include <geode/mesh/TriangleTopology.h>
namespace geode {

GEODE_CORE_EXPORT void
simplify_inplace(MutableTriangleTopology& mesh,
                 const FieldId<Vector<real,3>,VertexId> X_id, // X must be a field managed by the mesh so that it will be properly updated if topological simplification creates new vertices
                 const real distance,             // (Very) approximate distance between original and simplified
                 const real max_angle=pi/2,       // Max normal angle change in radians for one simplification step
                 const int min_vertices=-1,       // Stop if we simplify down to this many vertices (-1 for no limit)
                 const real boundary_distance=0); // How far we're allowed to move the boundary

GEODE_CORE_EXPORT Tuple<Ref<const TriangleTopology>,Field<const Vector<real,3>,VertexId>>
simplify(const TriangleTopology& mesh,
         RawField<const Vector<real,3>,VertexId> X,
         const real distance,             // (Very) approximate distance between original and simplified
         const real max_angle=pi/2,       // Max normal angle change in radians for one simplification step
         const int min_vertices=-1,       // Stop if we simplify down to this many vertices (-1 for no limit)
         const real boundary_distance=0);  // How far we're allowed to move the boundary


} // geode namespace