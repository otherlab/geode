#pragma once

#include <other/core/openmesh/TriMesh.h>

#ifdef USE_OPENMESH
namespace other {

// Decimate by collapsing edges, prioritized by quadric error, until no more
// collapsible edges are found. The angle error is the maximum allowed change
// normal of any of the faces, the max quadric error is a maximum allowed
// error (unit is length), roughly corresponding to maximum hausdorff distance,
// it will not create faces worse than min_face_quality, and it will not collapse
// edges on the boundary if a point would move more than the max quadric error.
// Returns the number of vertices collapsed.
OTHER_CORE_EXPORT int decimate(TriMesh &mesh,
                               int max_collapses,
                               double maxangleerror,
                               double maxquadricerror,
                               double min_face_quality);

}
#endif // USE_OPENMESH
