// Assorted utilities for inspecting properties of a mesh
#pragma once
#include <geode/mesh/TriangleTopology.h>
#include <geode/array/NestedField.h>

namespace geode {


// Count number of edges with length less than or equal to epsilon
int count_degenerate_edges(const TriangleTopology& mesh, const RawField<const Vector<real,3>, VertexId> X, const real epsilon);

// Count number of faces with area less than or equal to epsilon
int count_degenerate_faces(const TriangleTopology& mesh, const RawField<const Vector<real,3>, VertexId> X, const real epsilon);

// Returns volume of a closed mesh
real mesh_volume(const TriangleTopology& mesh, const RawField<const Vector<real,3>, VertexId> X);

// Returns volume of a component of a mesh defined by the faces in component_faces
// Assumes component_faces defines a closed sub-mesh
real mesh_volume(const TriangleTopology& mesh, const RawField<const Vector<real,3>, VertexId> X, const RawArray<const FaceId> component_faces);

// Returns clusters of connected faces
// Isolated vertices are ignored
NestedField<FaceId, ComponentId> get_component_faces(const TriangleTopology& mesh);

// Finds groups of vertices closer than epsilon to each other that aren't connected by edges
// This will detects duplicate vertices on a mesh that wouldn't be removed just by collapsing degenerate edges
// Returns each cluster of vertices that contains multiple unconnected components
Nested<VertexId> get_unconnected_clusters(const TriangleTopology& mesh, const RawField<const Vector<real,3>, VertexId> X, const real epsilon);

// Check if any two faces share the same three vertices
// In particular, this will detect degenerate pairs of faces such as {V0,V1,V2} and {V2,V1,V0}
bool has_duplicate_faces(const TriangleTopology& mesh);

} // geode namespace
