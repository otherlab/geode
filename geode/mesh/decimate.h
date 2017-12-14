// Quadric-based mesh decimation
#pragma once

#include <geode/mesh/TriangleTopology.h>
namespace geode {

GEODE_CORE_EXPORT Tuple<Ref<const TriangleTopology>,Field<const Vector<real,3>,VertexId>>
decimate(const TriangleTopology& mesh,
         RawField<const Vector<real,3>,VertexId> X,
         const real distance,             // (Very) approximate distance between original and decimation
         const real max_angle=pi/2,       // Max normal angle change in radians for one decimation step
         const int min_vertices=-1,       // Stop if we decimate down to this many vertices (-1 for no limit)
         const real boundary_distance=0); // How far we're allowed to move the boundary

GEODE_CORE_EXPORT void
decimate_inplace(MutableTriangleTopology& mesh,
                 RawField<const Vector<real,3>,VertexId> X, // This overload doesn't assume X is managed by mesh therefore must operate in PreserveTopology mode
                 const real distance,             // (Very) approximate distance between original and decimation
                 const real max_angle=pi/2,       // Max normal angle change in radians for one decimation step
                 const int min_vertices=-1,       // Stop if we decimate down to this many vertices (-1 for no limit)
                 const real boundary_distance=0); // How far we're allowed to move the boundary

GEODE_CORE_EXPORT Tuple<Ref<const TriangleTopology>,Field<const Vector<real,3>,VertexId>>
simplify_deprecated(const TriangleTopology& mesh,
         RawField<const Vector<real,3>,VertexId> X,
         const real distance,             // (Very) approximate distance between original and decimation
         const real max_angle=pi/2,       // Max normal angle change in radians for one decimation step
         const int min_vertices=-1,       // Stop if we decimate down to this many vertices (-1 for no limit)
         const real boundary_distance=0);  // How far we're allowed to move the boundary)

// As decimate_inplace, but will also try to remove topological noise by filling degenerate tunnels/tubes in the mesh
// Currently this will not merge disconnected components (though it may split components with only degenerate tubes connecting them)
// It also won't find things like coincident faces unless vertices are connected
// Positions are passed in as a FieldId to ensure X is managed by the MutableTriangleTopology (via add_field)
//   This is necessary since filling degenerating regions can create new vertices and X must be resized appropriately (though in most cases the final mesh should have fewer vertices than the input mesh) 
GEODE_CORE_EXPORT void
simplify_inplace_deprecated(MutableTriangleTopology& mesh,
                 const FieldId<Vector<real,3>,VertexId> X_id, // X must be a field managed by the mesh so that it will be properly updated if topological simplification creates new vertices
                 const real distance,             // (Very) approximate distance between original and decimation
                 const real max_angle=pi/2,       // Max normal angle change in radians for one decimation step
                 const int min_vertices=-1,       // Stop if we decimate down to this many vertices (-1 for no limit)
                 const real boundary_distance=0); // How far we're allowed to move the boundary

}
