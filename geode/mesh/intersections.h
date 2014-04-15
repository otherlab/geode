#pragma once

#include <geode/mesh/TriangleTopology.h>

namespace geode {

// low level interface

struct Intersection {

  enum {itEF, itFFF, itLoop} type;

  // best guess as to where this intersection actually is
  Vector<Interval, 3> construct;

  // vertices created for this intersection
  // for type ef: first edge, then face
  // for type fff: same order as face ids in data
  // for type loop: empty
  Array<VertexId> vertices;

  union {

    struct {
      HalfEdgeId edge; // we only use the smaller halfedge of any edge, so this would be the one stored here
      FaceId face;
      real t; // parameter along the edge (for sorting prior to triangulation TODO EXACT)
    } ef;

    struct {
      FaceId face1, face2, face3;
    } fff;

    struct {
      VertexId vertex;
    } loop;

  } data;

  // for loop vertices, this can be any (even) number, for triple-face intersections
  // it's exactly four, for regular intersections, exactly two (or the edge is on
  // the boundary, then exactly one).
  // We use an array, not a hashtable, because most intersections have only two elements.
  Array<int> connected_to;

  // operator for sorting of EF type intersections along their edge
  inline bool operator<(Intersection const &I) const {
    assert(I.type == itEF);
    return data.ef.t < I.data.ef.t;
  }
};

std::vector<Intersection> compute_edge_face_intersections(TriangleTopology const &mesh
                                                          Tuple<Ref<SimplexTree<Vector<real,3>,3>>, Array<FaceId>> face_tree,
                                                          Tuple<Ref<SimplexTree<Vector<real,3>,2>>, Array<HalfedgeId>> edge_tree);
std::vector<Intersection> compute_face_face_face_intersections(TriangleTopology const &mesh,
                                                               Tuple<Ref<SimplexTree<Vector<real,3>,3>>, Array<FaceId>> face_tree);

// return mappings from faces, edges, and vertices to all corresponding intersections
Tuple<Field<std::vector<int>, FaceId>,
      Field<std::vector<int>, HalfedgeId>,
      Field<int, VertexId>> assemble_paths(TriangleTopology const &mesh,
                                           Field<Vector<real,3>, VertexId> const &pos,
                                           std::vector<Intersection> &intersections);

void create_intersection_vertices(MutableTriangleTopology &mesh,
                                  FieldId<Vector<real,3>, VertexId> pos,
                                  std::vector<Intersection> const &intersections,
                                  FieldId<int,VertexId> vertex_intersections);

// return mappings from edges to corresponding edges (along intersection lines),
// and newly created faces to original faces
Tuple<FieldId<HalfedgeId, HalfedgeId>,
      FieldId<FaceId, FaceId>> retriangulate_faces(MutableTriangleTopology &mesh,
                                                   FieldId<Vector<real,3>, VertexId> pos,
                                                   std::vector<Intersection> const &intersections,
                                                   FieldId<std::vector<int>, FaceId> face_intersections,
                                                   FieldId<std::vector<int>, HalfedgeId> edge_intersections,
                                                   FieldId<int, VertexId> vertex_intersections);

Field<int, FaceId> compute_depths(TriangleTopology &mesh,
                                  Field<Vector<real,3>, VertexId> const &pos,
                                  Field<int, VertexId> const &vertex_intersections,
                                  Field<HalfedgeId, HalfedgeId> const &edge_correspondences,
                                  Tuple<Ref<SimplexTree<Vector<real,3>,3>>, Array<FaceId>> const &face_tree);

void stitch_meshes(MutableTriangleTopology &mesh,
                   FieldId<Vector<real,3>, VertexId> pos,
                   std::vector<Intersection> const &intersections,
                   FieldId<int, VertexId> vertex_intersections,
                   FieldId<HalfedgeId, HalfedgeId> edge_correspondences);


// high level interface
// this function does not perform any cleanup of the mesh, so the mesh still
// contains all deleted faces (originally deleted faces as well a faces deleted
// during CSG), and it returns the ID of a field containing a map to the original
// (now deleted) face for all faces created during CSG (a value of invalid means
// the face was already there).
FieldId<FaceId, FaceId> csg(MutableTriangleTopology &mesh, int depth, int pos_id_idx = vertex_position_id);

}
