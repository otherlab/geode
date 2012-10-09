// Set colors on a TriMesh

#include <other/core/openmesh/visualize.h>
namespace other {

template<> void visualize(TriMesh& mesh, const function<TriMesh::Color(VertexHandle)>& color) {
  if (!mesh.has_vertex_colors())
    mesh.request_vertex_colors();
  for (auto vi = mesh.vertices_sbegin(); vi != mesh.vertices_end(); ++vi)
    mesh.set_color(vi,color(vi));
}

template<> void visualize(TriMesh& mesh, const function<TriMesh::Color(EdgeHandle)>& color) {
  if (!mesh.has_edge_colors())
    mesh.request_edge_colors();
  for (auto ei = mesh.edges_sbegin(); ei != mesh.edges_end(); ++ei)
    mesh.set_color(ei,color(ei));
}

template<> void visualize(TriMesh& mesh, const function<TriMesh::Color(FaceHandle)>& color) {
  if (!mesh.has_face_colors())
    mesh.request_face_colors();
  for (auto fi = mesh.faces_sbegin(); fi != mesh.faces_end(); ++fi)
    mesh.set_color(fi,color(fi));
}

}
