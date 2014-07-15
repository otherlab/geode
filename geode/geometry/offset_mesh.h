// Mesh offsets
#pragma once

#include <geode/array/RawField.h>
#include <geode/mesh/TriangleTopology.h>
namespace geode {

// Below, we refer to this constant as alpha.
const real rough_offset_factor = 1/sqrt(5);

// Approximately offset a closed, convex mesh.  The result will lie within between distance alpha*offset
// and offset of the original mesh.  That is, the result is a guaranteed Theta(offset) offset.
GEODE_EXPORT Tuple<Ref<const TriangleTopology>,Field<const Vector<real,3>,VertexId>>
rough_offset_convex_mesh(const TriangleTopology& mesh,
                         RawField<const Vector<real,3>,VertexId> X,
                         const real offset);

/*
// Approximately offset a closed mesh treated as a volume.  The result will lie between the
// true offsets offset and 2*sqrt(3)*offset, which is sufficient when the goal is to avoid small errors.
// Returns mesh,X,new_to_old, where new_to_old maps new vertices to old vertices.
GEODE_EXPORT Tuple<Ref<const TriangleTopology>,Array<const Vector<real,3>>>
rough_offset_mesh(const TriangleTopology& mesh, RawField<const Vector<real,3>> X, const real offset);

// Approximately offset a mesh treated as a shell.  The mesh may have boundary.  The result will lie
// between the true offsets offset and 2*sqrt(3)*offset, which is sufficient when the goal is to avoid small errors.
// Returns mesh,X,new_to_old, where new_to_old maps new vertices to old vertices.
GEODE_EXPORT Tuple<Ref<const TriangleTopology>,Array<const Vector<real,3>>>
rough_offset_shell(const TriangleTopology& mesh, RawArray<const Vector<real,3>> X, const real offset);
*/

}
