#pragma once

#include <other/core/openmesh/TriMesh.h>

namespace other {

  void decimate(TriMesh &mesh, 
                int max_collapses = std::numeric_limits<int>::max(), 
                double maxangleerror = 90., 
                double maxquadricerror = std::numeric_limits<double>::infinity(), 
                double min_face_quality = 1e-5,
                double min_boundary_dot = .9999);

}
