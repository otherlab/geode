#pragma once

#include <other/core/openmesh/TriMesh.h>
#ifdef USE_OPENMESH
namespace other {

// smooth a mesh using bi-Laplacian smoothing with time constant t, all locked
// vertices are used as boundary conditions.
OTHER_CORE_EXPORT Ref<TriMesh> smooth_mesh(TriMesh &m, real t, real lambda, bool bilaplace, int val);

}
#endif // USE_OPENMESH
