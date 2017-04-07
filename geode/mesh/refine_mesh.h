#pragma once
#include <geode/mesh/TriangleTopology.h>

namespace other {
using namespace geode;

// Insert new vertices and flip edges to improve quality of mesh
//  * Edges in resulting mesh should be close to target_edge_length +/- about 30%
//  * Average valence of vertices should be close to 6/4 for internal/boundary vertices
void refine_mesh(MutableTriangleTopology& mesh, const FieldId<Vec2,VertexId> x_id,
                 const real target_edge_length, const int iterations=10);

}