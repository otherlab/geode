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
                 RawField<Vector<real,3>,VertexId> X,
                 const real distance,             // (Very) approximate distance between original and decimation
                 const real max_angle=pi/2,       // Max normal angle change in radians for one decimation step
                 const int min_vertices=-1,       // Stop if we decimate down to this many vertices (-1 for no limit)
                 const real boundary_distance=0); // How far we're allowed to move the boundary

}
