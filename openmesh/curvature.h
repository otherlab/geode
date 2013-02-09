// Measure the mean and Gaussian curvatures of meshes
#pragma once
#ifdef USE_OPENMESH

#include <other/core/openmesh/TriMesh.h>
namespace other {

OTHER_EXPORT Field<double,VertexHandle> mean_curvatures(const TriMesh& mesh);
OTHER_EXPORT Field<double,VertexHandle> gaussian_curvatures(const TriMesh& mesh);

}
#endif
