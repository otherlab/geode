// Measure the mean and Gaussian curvatures of meshes
#pragma once
#ifdef GEODE_OPENMESH

#include <geode/openmesh/TriMesh.h>
namespace geode {

GEODE_CORE_EXPORT Field<double,VertexHandle> mean_curvatures(const TriMesh& mesh);
GEODE_CORE_EXPORT Field<double,VertexHandle> gaussian_curvatures(const TriMesh& mesh);

}
#endif
