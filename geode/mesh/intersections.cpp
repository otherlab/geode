#include <geode/mesh/intersections.h>
#include <geode/utility/stl.h>

namespace geode {

/*

= Intersections

- compute all edge-face intersections, and store with affected face and
edge (ignore those that share a common vertex)

*/

typedef Vector<real,3> TV;

struct edge_face_intersection_helper {
  TriangleTopology const &mesh;
  Tuple<Ref<SimplexTree<TV,3>>, Array<FaceId>> face_tree;
  Tuple<Ref<SimplexTree<TV,2>>, Array<HalfedgeId>> edge_tree;

  std::vector<Intersection> intersections;

  edge_face_intersection_helper(TriangleTopology const &mesh,
                                Tuple<Ref<SimplexTree<TV,3>>, Array<FaceId>> face_tree,
                                Tuple<Ref<SimplexTree<TV,2>>, Array<HalfedgeId>> edge_tree)
  : mesh(mesh), face_tree(face_tree), edge_tree(edge_tree) {}

  inline bool cull(int nf, int ne) const {
    return false;
  }

  void check_intersection(int nf, int ne) {
    // check if face nf and edge ne intersect, and add to intersections if they do
    HalfedgeId h = edge_tree.y[ne];
    FaceId f = face_tree.y[nf];

    // don't check incident edge/face pairs
    if (mesh.common_vertex(f, h).valid()) {
      return;
    }

    Segment<TV> const &segment = edge_tree.x->simplices[ne];
    Ray<TV> ray(segment);
    Triangle<TV> const &face = face_tree.x->simplices[nf];

    // TODO EXACT
    if (face.intersection(ray)) {
      Intersection intersection;
      intersection.type = itEF;
      // make sure that the edge we store is the internal halfedge with the smallest index
      if (mesh.is_boundary(h))
        intersection.data.ef.edge = reverse(h);
      else
        intersection.data.ef.edge = h;
      intersection.data.ef.face = f;
      intersection.data.ef.t = ray.t_max;
      intersection.construct = ray.point(ray.t_max);
      intersections.push_back(intersection);
    }
  }

  void leaf(int nf, int ne) {
    auto es = edge_tree.x->prims(ne);
    auto fs = face_tree.x->prims(nf);

    for (int ei : es) {
      for (int fi: fs) {
        check_intersection(fi, ei);
      }
    }
  }
};

std::vector<Intersection> compute_edge_face_intersections(TriangleTopology const &mesh,
                                                          Tuple<Ref<SimplexTree<TV,3>>, Array<FaceId>> face_tree,
                                                          Tuple<Ref<SimplexTree<TV,2>>, Array<HalfedgeId>> edge_tree) {
  edge_face_intersection_helper helper(mesh, face_tree, edge_tree);
  double_traverse(face_tree, edge_tree, helper);
  return helper.intersections;
}

/*

- compute all face-face-face intersections, and store with affected faces

*/

struct face_face_intersection_helper {
  TriangleTopology const &mesh;
  Tuple<Ref<SimplexTree<TV,3>>, Array<FaceId>> face_tree;

  // TODO EXACT: we store constructed edges, but they're only necessary for tree
  // creation/lookup, so giving the second traversal a thickness large enough to
  // contain the largest interval should be safe and easy.
  Array<Vector<int,2>> indices;
  Array<TV> vertices;
  Array<Tuple<FaceId, FaceId>> faces;

  face_face_intersection_helper(TriangleTopology const &mesh,
                                Tuple<Ref<SimplexTree<TV,3>>, Array<FaceId>> face_tree)
  : mesh(mesh), face_tree(face_tree) {}

  inline bool cull(int nf, int ne) const {
    return false;
  }

  void check_intersection(int nf1, int nf2) {
    // check if faces intersect, and add to generated edges if they do
    FaceId f1 = face_tree.y[nf1];
    FaceId f2 = face_tree.y[nf2];

    // don't check incident faces
    if (mesh.common_halfedge(f1, f2).valid()) {
      return;
    }

    Triangle<TV> const &face1 = face_tree.x->simplices[nf1];
    Triangle<TV> const &face2 = face_tree.x->simplices[nf2];
    Segment<TV> result;

    // TODO EXACT (not super-necessary to do exact if this is made conservative)
    if (face1.intersection(face2, result)) {
      indices.append(vec(vertices.size(), vertices.size()+1));
      vertices.append(result.x0);
      vertices.append(result.x1);
      faces.append(tuple(face1, face2));
    }
  }

  void leaf(int nf1, int nf2) {
    auto fs1 = face_tree.x->prims(nf1);
    auto fs2 = face_tree.x->prims(nf2);

    for (int fi : fs1) {
      for (int fj: fs2) {
        check_intersection(fi, fj);
      }
    }
  }
};

struct face_face_face_intersection_helper {
  TriangleTopology const &mesh;

  Ref<SimplexTree<TV,2>> edge_tree;
  Array<Tuple<FaceId,FaceId>> const &edge_faces;

  Tuple<Ref<SimplexTree<TV,3>>, Array<FaceId>> face_tree;

  Hashtable<Vector<FaceId,3>> checked;

  std::vector<Intersection> intersections;

  face_face_face_intersection_helper(TriangleTopology const &mesh,
                                     Ref<SimplexTree<TV,2>> edge_tree,
                                     Array<Tuple<FaceId,FaceId>> const &edge_faces,
                                     Tuple<Ref<SimplexTree<TV,3>>, Array<FaceId>> face_tree)
  : mesh(mesh), edge_tree(edge_tree), edge_faces(edge_faces), face_tree(face_tree) {}

  inline bool cull(int nf, int ne) const {
    return false;
  }

  void check_intersection(int ne, int nf) {
    Vector<FaceId,3> f = vec(edge_faces[ne].x, edge_faces[ne].y, face_tree.y[nf]);
    small_sort(f.x, f.y, f.z);

    // check if already seen
    if (checked.contains(f)) {
      return;
    } else {
      checked.insert(f);
    }

    // no two may have an edge in common
    if (mesh.common_halfedge(f.x,f.y).valid() ||
        mesh.common_halfedge(f.x,f.z).valid() ||
        mesh.common_halfedge(f.y,f.z).valid())
      return;

    // not all of them may have a single vertex in common
    auto cv12 = mesh.common_vertex(f.x,f.y);
    if (cv12.valid && mesh.common_vertex(f.y,f.z) == cv12)
      return;

    // TODO EXACT using the three input faces
    Segment<TV> const &edge = edge_tree.simplices[ne];
    Ray<TV> ray(segment);
    Triangle<TV> const &face = face_tree.x->simplices[nf];

    if (face.intersection(ray)) {
      Intersection intersection;
      intersection.type = itFFF;
      intersection.data.fff.face1 = f.x;
      intersection.data.fff.face2 = f.y;
      intersection.data.fff.face3 = f.z;
      intersection.construct = ray.point(ray.t_max);
      intersections.push_back(intersection);
    }
  }

  void leaf(int nf, int ne) {
    auto es = edge_tree.x->prims(ne);
    auto fs = face_tree.x->prims(nf);

    for (int ei : es) {
      for (int fi: fs) {
        check_intersection(fi, ei);
      }
    }
  }
};

std::vector<Intersection> compute_face_face_face_intersections(TriangleTopology const &mesh,
                                                               Tuple<Ref<SimplexTree<TV,3>>, Array<FaceId>> face_tree) {
  face_face_intersection_helper ff_helper(mesh, face_tree);
  double_traverse(face_tree, ff_helper);
  auto edge_tree = new_<SimplexTree<TV,2>>(SegmentSoup(ff_helper.indices, ff_helper.vertices));
  face_face_face_intersection_helper fff_helper(mesh, edge_tree, ff_helper.faces, face_tree);
  double_traverse(face_tree, edge_tree, fff_helper);
  return fff_helper.intersections;
}

/*

= Paths

- find loop vertices: a vertex incident to a face f0 and another face f1 incident to an
  edge intersecting f0 is a loop vertex.

- (debug only) make sure intersections are symmetric. For any
intersection between a face f and an edge e, one of the following must
hold for all faces fi incident on e:
  * fi intersects one of the edges of f
  * fi and f share a loop vertex
  * another edge e2 != e of fi intersects f

- find paths: fill in connected_to on all intersections

- insert face-face-face intersections into paths in the correct order

*/

Tuple<Field<std::vector<int>, FaceId>,
      Field<std::vector<int>, HalfedgeId>,
      Field<int, VertexId>> assemble_paths(TriangleTopology const &mesh,
                                           Field<TV, VertexId> const &pos,
                                           std::vector<Intersection> &intersections) {

  const bool debug = false;

  // make fields compatible with the mesh
  Field<std::vector<int>, FaceId> face_intersections = mesh.create_compatible_field<std::vector<int>, FaceId>();
  Field<std::vector<int>, HalfedgeId> edge_intersections = mesh.create_compatible_field<std::vector<int>, EdgeId>();
  Field<int, VertexId> vertex_intersections = mesh.create_compatible_field(int, VertexId);
  for (auto &f : vertex_intersections.flat) {
    f = -1;
  }

  // register regular intersections in fields
  for (int i = 0; i < intersections.size(); ++i) {
    Intersection const &I = intersections[i];

    switch (I.type) {
      case itEF: {
        face_intersections[I.data.ef.face].push_back(i);
        edge_intersections[I.data.ef.edge].push_back(i); // this is always the internal halfedge with smallest index
        break;
      }
      case itFFF:
        break;
      case itLoop:
      default:
        GEODE_UNREACHABLE();
    }
  }

  // Use this to look up edge intersections (takes care of looking up by lower index halfedge)
  struct GetEdgeIntersections {
    TriangleTopology const &mesh;
    Field<std::vector<int>, HalfedgeId> &edge_intersections;

    GetEdgeIntersections(TriangleTopology const &mesh, Field<std::vector<int>, HalfedgeId> const &ei)
    : mesh(mesh), edge_intersections(ei) {};

    std::vector<int> &operator[](HalfedgeId id) {
      HalfedgeId r = mesh.reverse(id);
      if (!mesh.is_boundary(r) && r < id)
        id = r;
      return edge_intersections[id];
    }

  } get_edge_intersections(mesh, edge_intersections);

  // walk paths and fill in connected_to, creating loop vertices as we go
  for (int i = 0; i < intersections.size(); ++i) {
    if (intersections[i].type == itEF) {
        auto const &f = intersections[i].face;
        auto const &e = intersections[i].edge;
        // for all incident faces to the edge, find the next intersection and connect them
        for (auto fi : mesh.faces(e)) {
          // the edge is a boundary edge, nothing to do here
          if (!f1.valid())
            continue;

          bool found = false;

          // fi intersects one of the edges of f (debug only: check that it's exactly one)
          for (auto ei : mesh.halfedges(f)) {
            for (int i2 : get_edge_intersections[ei]) {
              Intersection const &I2 = intersections[i2];

              if (I2.face == fi) {
                OTHER_ASSERT(!found); // fi cannot intersect more than one edge of f since one of its edges intersects f
                found = true;
                intersections[i].connected_to.append(i2);
                if (!debug)
                  break;
              }
            }

            if (!debug && found)
              break;
          }

          if (!debug && found)
            continue;

          // another edge ei != e of fi intersects f (debug only: check that it's exactly one)
          for (auto ei : mesh.halfedges(fi)) {
            if (ei == e || ei == reverse(e))
              continue;
            for (int i2 : get_edge_intersections[ei]) {
              Intersection const &I2 = intersections[i2];

              if (I2.face == f) {
                OTHER_ASSERT(!found); // only a maximum of two edges of fi can intersect f
                found = true;
                intersections[i].connected_to.append(i2);
                if (!debug)
                  break;
              }
            }

            if (!debug && found)
              break;
          }

          if (!debug && found)
            continue;

          // fi and f share a loop vertex
          auto lv = mesh.common_vertex(fi, f);
          if (lv.valid()) {
            if (vertex_intersections[lv] == -1) {
              // add an intersection for the loop vertex
              vertex_intersections[lv] = intersections.size();
              Intersection intersection;
              intersection.type = itLoop;
              intersection.construct = pos[lv];
              intersection.data.loop.vertex = lv;
              intersections.push_back(intersection);
            }

            intersections[i].connected_to.append(vertex_intersections[lv]);
            intersections[vertex_intersections[lv]].connected_to.append(i);

            continue;
          }

          // otherwise, this must be an error.
          OTHER_ASSERT(false);
        }

    } else if (I.type == itLoop) {
      // all loop vertices are at the end, they should already be connected, quit the loop
      break;
    } else {
      // there should not be anything else
      GEODE_UNREACHABLE();
    }
  }

  // make a map from face pairs to face-face-face intersections
  Hashtable<Tuple<FaceId, FaceId>, std::vector<Tuple<int, FaceId>>> fff_by_edge;
  for (int i = 0; i < intersections.size(); ++i) {
    Intersection const &I = intersections[i];

    switch (I.type) {
      case itEF:
        break;
      case itFFF: {
        Tuple<FaceId, FaceId> p1(I.data.fff.face1, I.data.fff.face2);
        Tuple<FaceId, FaceId> p2(I.data.fff.face1, I.data.fff.face3);
        Tuple<FaceId, FaceId> p3(I.data.fff.face2, I.data.fff.face3);

        fff_by_edge[p1].push_back(tuple(i, I.data.fff.face3));
        fff_by_edge[p2].push_back(tuple(i, I.data.fff.face2));
        fff_by_edge[p3].push_back(tuple(i, I.data.fff.face1));
        break;
      }
      case itLoop:
        // we can quit, once we're in loop territory that's all we'll see
        i = intersections.size();
        break;
      default:
        GEODE_UNREACHABLE();
    }
  }

  // insert face-face-face intersections into the segments they belong to, sorted

  for (auto &it : fff_by_edge) {
    // sort the fff intersections along each face pair
    Tuple<FaceId, FaceId> const &fp = it.key();
    std::vector<Tuple<int,FaceId>> const &ints = it.data();

    // find the regular intersections corresponding to the intersection between
    // these two faces
    int end1 = -1, end2 = -1;

    // do fp.x and fp.y share a loop vertex?
    VertexId cv = mesh.common_vertex(fp.x, fp.y);
    if (cv.valid()) {
      end1 = vertex_intersections[cv];
    }

    // does any edge of fp.x have an intersection with fp.y?
    for (HalfedgeId ei : halfedges(fp.x)) {
      for (int eii : get_edge_intersections[ei]) {
        if (intersections[eii].data.ef.face == fp.y) {
          if (end1 == -1) {
            end1 = eii;
          } else {
            OTHER_ASSERT(end2 == -1);
            end2 = eii;
            if (!debug)
              goto found_all;
          }
        }
      }
    }

    // does any edge of fp.y have an intersection with fp.x?
    for (HalfedgeId ei : halfedges(fp.y)) {
      for (int eii : get_edge_intersections[ei]) {
        if (intersections[eii].data.ef.face == fp.x) {
          if (end1 == -1) {
            end1 = eii;
          } else {
            OTHER_ASSERT(end2 == -1);
            end2 = eii;
            if (!debug)
              goto found_all;
          }
        }
      }
    }

    // now, we ought to have collected exactly two intersections.
    GEODE_ASSERT(end1 != -1 && end2 != -1);

    found_all:

    // remove the path edge between the end points
    OTHER_ASSERT(intersections[end1].connected_to.contains(end2));
    OTHER_ASSERT(intersections[end2].connected_to.contains(end1));

    intersections[end1].connected_to.remove_first_lazy(end2);
    intersections[end2].connected_to.remove_first_lazy(end1);


    if (debug || ints.size() > 1) {
      // TODO EXACT
      // compute the edge implied by the face pair
      TV N = cross(mesh.triangle(fp.x, pos).normal(),
                   mesh.triangle(fp.y, pos).normal());

      std::vector<Prioritize<int>> sorted_ints;
      for (auto I : ints) {
        real d = dot(N, intersections[I.x].construct);
        sorted_ints.push_back(prioritize(I.x, d));
      }

      // add the end points
      real d1 = dot(N, intersections[end1].construct);
      real d2 = dot(N, intersections[end2].construct);
      sorted_ints.push_back(prioritize(end1, d1));
      sorted_ints.push_back(prioritize(end2, d2));

      // sort all intersections (smallest first)
      fallback_sort(sorted_ints, std::less<Prioritize<int>>());

      // make sure the end points are in fact at the ends
      OTHER_ASSERT(sorted_ints.front().a == end1 || sorted_ints.front().a == end2);
      OTHER_ASSERT(sorted_ints.back().a == end1 || sorted_ints.back().a == end2);

      // insert the sorted string of face-face-face intersections into the path
      // between the end points
      for (int ii = 0; ii < sorted_ints.size()-1; ++ii) {
        int i1 = sorted_ints[ii].a;
        int i2 = sorted_ints[ii+1].a;

        OTHER_ASSERT(!intersections[i1].connected_to.contains(i2));
        OTHER_ASSERT(!intersections[i2].connected_to.contains(i1));
        intersections[i1].connected_to.append(i2);
        intersections[i2].connected_to.append(i1);
      }

    } else {

      // only one intersection, insert between the end point intersections
      intersections[end1].connected_to.append(ints[0].x);
      intersections[end2].connected_to.append(ints[0].x);
      intersections[ints[0].x].connected_to.append(end1);
      intersections[ints[0].x].connected_to.append(end2);

    }
  }

  // register face-face-face intersections on the faces they belong to
  for (int i = 0; i < intersections.size(); ++i) {
    Intersection const &I = intersections[i];

    switch (I.type) {
      case itEF:
        break;
      case itFFF: {
        face_intersections[I.data.fff.face1].push_back(i);
        face_intersections[I.data.fff.face2].push_back(i);
        face_intersections[I.data.fff.face3].push_back(i);
        break;
      }
      case itLoop:
        // we can quit, once we're in loop territory that's all we'll see
        i = intersections.size();
        break;
      default:
        GEODE_UNREACHABLE();
    }
  }

  // debug only: make sure all connections are symmetric and intersections have
  // the correct number of connections
  if (debug) {
    for (int i = 0; i < intersections.size(); ++i) {
      Intersection const &I = intersections[i];

      OTHER_ASSERT(I.connected_to.is_unique());

      switch (I.type) {
        case itEF: {
          OTHER_ASSERT(mesh.is_boundary(I.edge) && I.connected_to.size() == 1 ||
                       !mesh.is_boundary(I.edge) && I.connected_to.size() == 2);
          break;
        }
        case itFFF: {
          OTHER_ASSERT(I.connected_to.size() == 6);
          break;
        }
        case itLoop: {
          OTHER_ASSERT(I.connected_to.size() % 2 == 0);
          break;
        }
      }

      for (int i2 : I.connected_to) {
        Intersection const &I2 = intersections[i2];
        OTHER_ASSERT(I2.connected_to.contains(i));
      }
    }
  }

  return tuple(face_intersections, edge_intersections, vertex_intersections);
}

/*

= Retriangulation

- requires pos to be a field registered in the mesh (hence the id), so resizing, etc.
  is properly taken care of.

- for each intersection, add as many vertices as necessary (2 for regular,
  3 for face-face-face intersections) to make the mesh manifold, and
  associate the new vertices to the intersection
- update vertex field storing intersection ids per vertex (could already have loop vertex data in it)

*/

void create_intersection_vertices(MutableTriangleTopology &mesh, FieldId<TV, VertexId> pos,
                                  std::vector<Intersection> const &intersections,
                                  FieldId<int,VertexId> vertex_intersections_id) {
  Field<int,VertexId> vertex_intersections = mesh.field(vertex_intersections_id);
  for (int i = 0; i < intersections.size(); ++i) {
    Intersection &I = intersections[i];
    OTHER_ASSERT(I.vertices.empty());
    switch (I.type) {
      case itEF:
        I.vertices.extend(vec(mesh.add_vertex(), mesh.add_vertex()));
        break;
      case itFFF:
        I.vertices.extend(vec(mesh.add_vertex(), mesh.add_vertex(), mesh.add_vertex()));        mesh.field(pos)[I.vertices[0]]
        break;
      case itLoop:
        // we may have to duplicate loop vertices, but not for triangulation,
        // this will only become necessary during stitching, so we'll do that later
        break;
      default:
        GEODE_UNREACHABLE();
    }

    for (auto vi : I.vertices) {
      vertex_intersections[vi] = i;
      mesh.field(pos)[vi] = I.construct;
    }
  }
}

/*

- retriangulate faces: for each face
  * look up paths on the face to use as constraints for triangulation
  * compute triangulation for the face that includes all new vertices, and
    respects all constraints

- replace old faces with retriangulation

- return correspondences of intersection path edges (store halfedge id with
  consistent ordering that's not our own reverse) and original face id for each
  newly created face.

*/

Tuple<FieldId<HalfedgeId, HalfedgeId>,
      FieldId<FaceId, FaceId>> retriangulate_faces(MutableTriangleTopology &mesh,
                                                   FieldId<TV, VertexId> pos_id,
                                                   std::vector<Intersection> const &intersections,
                                                   FieldId<std::vector<int>, FaceId> face_intersections_id,
                                                   FieldId<std::vector<int>, HalfedgeId> edge_intersections_id,
                                                   FieldId<int, VertexId> vertex_intersections_id) {

  auto pos = mesh.field(pos_id);
  auto vertex_intersections = mesh.field(vertex_intersections_id);
  auto edge_intersections = mesh.field(edge_intersections_id);
  auto face_intersections = mesh.field(face_intersections_id);

  // sort all edge intersections for t value along their edges
  // TODO EXACT
  for (auto he : mesh.interior_halfedges()) {
    fallback_sort(edge_intersections[he], std::less<Intersection>());
  }

  auto edge_correspondences_id = mesh.add_halfedge_field<HalfedgeId>();
  auto face_correspondences_id = mesh.add_face_field<FaceId>();
  auto edge_correspondences = mesh.field(edge_correspondences_id);
  auto face_correspondences = mesh.field(face_correspondences_id);

  edge_correspondences.flat.fill(HalfedgeId())
  face_correspondences.flat.fill(Face())

  // Map from a sorted pair of intersections to pairs of vertices. Each pair of vertices
  // is sorted the same (vertex corresponding to lower index intersection first).
  // There should be zero or two vertex pairs for any given intersection pair.
  Hashtable<Vector<int,2>, Array<Vector<VertexId,2>>> intersection_edges;

  for (auto const &f : mesh.faces()) {

    // make a set of vertices and edge constraints to use in triangulation
    Array<Vector<int,2>> edges;
    Array<VertexId> vertices;
    Hashtable<int, int> intersection_to_local_idx; // mapping from intersection index to local index (into vertices)

    // add the original triangle vertices
    vertices.extend(mesh.vertices(f));

    // add intersection index translation for loop vertices
    for (int i = 0; i < 3; ++i) {
      int iid = vertex_intersections[vertices[i]];
      if (iid != -1)
        intersection_to_local_idx[iid] = i;
    }

    // each constraint is an intersection edge, or part of the triangle boundary

    // add boundary vertices and edges
    for (int i = 0; i < 3; ++i) {
      HalfedgeId he = mesh.halfedge(f, i);
      VertexId startv = mesh.vertex(f, i);
      assert(mesh.src(he) == startv);
      assert(startv == vertices[i]);
      VertexId endv = mesh.dst(he);

      // check where the intersections are stored
      std::vector<int> ints;
      if (mesh.is_boundary(mesh.reverse(he)) || he < mesh.reverse(he)) {
        // they're with he
        ints = edge_intersections[he];
      } else {
        // they're with reverse(he)
        ints = edge_intersections[mesh.reverse(he)];
        std::reverse(ints.begin(), ints.end());
      }

      // add all vertices contained in these constraints
      for (int ii = 0; ii < ints.size(); ++ii) {
        // all these intersections intersect our edge
        OTHER_ASSERT(intersections[ints[ii]].type == itEF);
        OTHER_ASSERT(intersections[ints[ii]].data.ef.edge == he || intersections[ints[ii]].data.ef.edge == reverse(he));
        intersection_to_local_idx[ii] = vertices.size();
        // first vertex is the one dedicated to the edge
        vertices.push_back(intersections[ints[ii]].vertices[0]);
      }

      // add triangulation constraints along this edge
      if (!ints.empty()) {
        edges.push_back(vec(i, intersection_to_local_idx[ints.front()));
        for (int j = 1; j < ints.size(); ++j) {
          edges.push_back(vec(intersection_to_local_idx[ints[j-1]],
                              intersection_to_local_idx[ints[j]]))
        }
        edges.push_back(vec(intersection_to_local_idx[ints.back()], (i+1)%3));
      } else {
        edges.push_back(vec(i, (i+1)%3));
      }
    }

    // make all interior vertices and interior edge constraints (using
    // connected_to).

    for (int i : face_intersections[f]) {
      Intersection const &I = intersections[i];
      intersection_to_local_idx[i] = vertices.size();

      // get the vertex for this intersection and add it to the set of vertices
      // to triangulate
      switch (I.type) {
        case itEF:
          // we must be the face in this intersection
          OTHER_ASSERT(I.data.ef.face == f);
          // first vertex is the one dedicated to the edge, second one to the face
          vertices.push_back(I.vertices[1]);
          break;
        case itFFF:
          if (I.data.fff.face1 == f) {
            vertices.push_back(I.vertices[0]);
          } else if (I.data.fff.face2 == f) {
            vertices.push_back(I.vertices[1]);
          } else if (I.data.fff.face3 == f) {
            vertices.push_back(I.vertices[2]);
          } else {
            OTHER_FATAL_ERROR();
          }
          break;
        case itLoop:
        default:
          OTHER_UNREACHABLE();
      }

      // go through all connected intersections, and remember edges to vertices
      // that are already there.
      for (int i2 : I.connected_to) {
        if (!intersection_to_local_idx.contains(i2))
          continue; // we will add this edge when we add i2's vertex

        edges.append(vec(intersection_to_local_idx[i], intersection_to_local_idx[i2]));

        // Remember vertex pairs of intersection edges for later reconstruction of
        // edge correspondence
        if (i < i2) {
          intersection_edges[vec(i,i2)].append(vec(vertices[intersection_to_local_idx[i]],
                                                   vertices[intersection_to_local_idx[i2]]));
        } else {
          intersection_edges[vec(i2,i)].append(vec(vertices[intersection_to_local_idx[i2]],
                                                   vertices[intersection_to_local_idx[i]]));
        }
      }
    }

    // TODO EXACT: this should be replaced by an exact non-delaunay version of
    // recursive triangulation
    // project points into the face plane so delaunay can deal with them
    Array<Vector<real,2>> projected;
    // make a Frame such that applying the frame rotates the face normal to z
    auto rot = Rotation<TV>::from_rotated_vector(mesh.normal(f), vec(0, 0, 1));
    for (auto v : vertices) {
      projected.push_back((rot*pos[v]).xy());
    }
    Ref<TriangleTopology> faces = delaunay_points(projected, edges);

    // replace original face with new faces
    mesh.erase(f, false);
    for (auto newf : faces.faces()) {}
      Vector<VertexId,3> verts = faces.vertices(newf);
      auto added = mesh.add_face(vertices[verts[0].id],
                                 vertices[verts[1].id],
                                 vertices[verts[2].id]);

      // fill face_correspondences
      face_correspondences[added] = f;
    }
  }

  // make sure all intersection edges have a valid number of vertex representatives
  for (auto p : intersection_edges) {
    OTHER_ASSERT(p.value().size() == 2);

    // get the edges corresponding to the two vertex pairs
    HalfedgeId he1 = mesh.halfedge(p.value().front().x, p.value().front().y);
    HalfedgeId he2 = mesh.halfedge(p.value().back().x, p.value().back().y);

    // the two halfedges are aligned, add four edge_correspondences entries
    edge_correspondences[he1] = mesh.reverse(he2);
    edge_correspondences[he2] = mesh.reverse(he1);
    edge_correspondences[mesh.reverse(he1)] = he2;
    edge_correspondences[mesh.reverse(he2)] = he1;
  }

  return tuple(edge_correspondences_id, face_correspondences_id);
}

/*

= Depth

- Assign a depth to each face
  (closed meshes contained in the given tree are used for computation of depth)
  * Shoot a ray from infinity, targeted at a non-constructed vertex
  * use the original triangles for all intersection tests (use the original face tree)
  * Mark the surface hit first as depth b (where b is the winding number at
    infinity, as determined by counting nested negative shapes)
  * Propagate the depth to this connected component: floodfill until a path is
    reached. Propagate across the path using topological orientation.
  * Remove the completed component from consideration when computing raycasts,
    and adjust the winding number at infinity if necessary.

*/

Field<int, FaceId> compute_depths(TriangleTopology const &mesh,
                                  Field<TV, VertexId> const &pos,
                                  Field<int, VertexId> const &vertex_intersections,
                                  Field<HalfedgeId, HalfedgeId> const &edge_correspondences,
                                  Tuple<Ref<SimplexTree<TV,3>>, Array<FaceId>> const &face_tree) {

  Field<int, FaceId> depth_map = mesh.create_compatible_field<int,FaceId>();
  Field<bool, VertexId> depth_assigned = mesh.create_compatible_field<bool,VertexId>();

  for (auto vi : mesh.vertices()) {
    // find an original vertex (vertex_intersections[vi] == -1)
    if (depth_assigned[vi] || vertex_intersections[vi] != -1)
      continue;

    // compute the depth of this vertex
    // TODO EXACT
    // this simple normal is not safe -- the normal must point "outside"
    Ray<TV> ray(pos[vi], mesh.normal(vi, pos), true);
    Array<Ray<TV> intersections = face_tree.x->intersections(ray, 0);

    int depth = 0;
    for (auto r : intersections) {
      auto f = face_tree.y[r.aggregate_id];

      if (mesh.vertices(f).contains(vi));
        continue;

      auto simplex = face_tree.simplices[r.aggregate_id];
      if (dot(r.direction, simplex.normal()) > 0) {
        depth++;
      } else {
        depth--;
      }
    }

    // propagate the depth across this connected component
    std::vector<FaceId> upnext();

    // initialize any face incident to vi (be wary of boundaries)
    FaceId fi = mesh.face(mesh.halfedge(vi));
    if (!fi.valid())
      fi = mesh.face(mesh.reverse(mesh.halfedge(vi)));
    OTHER_ASSERT(fi.valid());
    OTHER_ASSERT(!depth_assigned[fi]);
    depth_assigned[fi] = true;
    depth_map[fi] = depth;
    upnext.push_back(fi);

    while (!upnext.empty()) {
      FaceId id = upnext.back();
      depth = depth_map[id];
      upnext.pop_back();

      for (auto he : mesh.halfedges(id)) {
        HalfedgeId intersection_edge = edge_correspondences[he];

        #define SET_DEPTH(halfedge, depth) \
          FaceId f = mesh.face(halfedge); \
          if (depth_assigned[f]) \
            OTHER_ASSERT((depth) == depth_map[f]); \
          else { \
            depth_assigned[f] = true; \
            depth_map[f] = (depth); \
            upnext.push_back(f); \
          }

        if (intersection_edge.valid()) {
          // propagate to next bit of the same surface
          SET_DEPTH(intersection_edge, depth);

          // is the other surface deeper or less deep?
          // TODO EXACT
          int diff = mesh.triangle(mesh.face(intersection_edge), pos).phi(pos[mesh.opposite(he)]) > 0 ? 1 : -1;

          // propagate to other surface
          HalfedgeId r = mesh.reverse(he);
          HalfedgeId rie = edge_correspondences[r];
          OTHER_ASSERT(rie.valid());
          SET_DEPTH(r, depth + diff);
          SET_DEPTH(rie, depth + diff);

        } else {
          // propagate across this edge
          HalfedgeId r = mesh.reverse(he);
          if (r.valid())
            SET_DEPTH(r, depth);
        }
      }
    }
  }

  return depth_map;
}

/*

= Stitching

- Replace all intersection vertices with a single representative (this should
  not lead to non-manifold edges any more)
- Resolve non-manifold vertices by duplication

*/

void stitch_meshes(MutableTriangleTopology &mesh,
                   FieldId<TV, VertexId> pos_id,
                   std::vector<Intersection> const &intersections,
                   FieldId<int, VertexId> vertex_intersections_id,
                   FieldId<HalfedgeId, HalfedgeId> edge_correspondences_id) {

  auto pos = mesh.field(pos_id);
  auto vertex_intersections = mesh.field(vertex_intersections_id);
  auto edge_correspondences = mesh.field(edge_correspondences_id);

  // loop vertices may have to be split, they may be attached to several connected
  // components in the new mesh. Count the number of components around the loop
  // vertex first, split the vertex and redistribute the incident faces before
  // merging intersection edges.
  for (auto vi : mesh.vertices()) {
    int iid = vertex_intersections[vi];
    if (iid == -1)
      continue;
    if (intersections[iid].type != itLoop)
      continue;

    intersections[iid].vertices.append(vi);

    // can't make more than one component with only two edges
    if (intersections[iid].connected_to.size() <= 2)
      continue;

    while (true) {
      // compute the component starting at first halfedge (which is a boundary
      // halfedge)
      HalfedgeId he = mesh.halfedge(vi);
      std::vector<HalfedgeId> component(1, he);
      OTHER_ASSERT(mesh.is_boundary(component.front()));

      // walk right (in the direction of where the faces connected to our halfedge
      // are), and add faces to our component until we have closed the loop
      while (true) {
        HalfedgeId ohe = mesh.reverse(he);
        if (mesh.face(ohe).valid()) {
          // next one is connected via a face
          HalfedgeId next_he = mesh.next(ohe);
          // can't finish with a face, since the first he was a boundary
          OTHER_ASSERT(next_he != component[0]);
          component.push_back(next_he);
          he = next_he;
        } else {
          // no face here, check what this connects to
          HalfedgeId next_he = edge_correspondences[he];
          if (!next_he.valid()) {
            // there was a boundary here initially, we're done with this component
            break;
          } else {
            next_he = mesh.reverse(next_he);
            OTHER_ASSERT(mesh.is_boundary(next_he));
            OTHER_ASSERT(vertex_intersections[mesh.dst(he)] == vertex_intersections[mesh.dst(next_he)]);
            if (next_he == component[0]) {
              // we're done!
              break;
            } else {
              component.push_back(next_he);
              he = next_he;
            }
          }
        }
      }

      // check if this component is all the outgoing edge there are (then we're good here)
      if (component.size() == stl::distance(mesh.outgoing(vi).begin(), mesh.outgoing(vi).end())) {
        break;
      }

      // There's more to this vertex than meets the eye.

      // Make a new vertex newv as a copy of vi
      // (note that vi and the loop iterator remains valid because the iterators only store ids)
      auto newv = mesh.add_vertex();
      pos[newv] = pos[vi];
      vertex_intersections[newv] = vertex_intersections[vi];
      intersections[iid].vertices.append(newv);

      int i = 0;
      HalfedgeId he = component.front();

      // store the halfedge that is left dangling when we take this section out
      // of the loop
      HalfedgeId dangling = mesh.prev(he);

      do {

        // he is always a boundary halfedge here
        OTHER_ASSERT(mesh.is_boundary(he));

        // set the src vertex on the outgoing boundary edges
        unsafe_set_src(he, newv);

        // move on to the next (which cannot be past the end, and cannot be a boundary)
        ++i;
        OTHER_ASSERT(i < component.size());
        he = component[i];
        OTHER_ASSERT(!mesh.is_boundary(he));

        // replace vi with newv in all faces in this section
        do {
          bool changed = unsafe_replace_vertex(mesh.face(he), vi, newv);
          OTHER_ASSERT(changed);

          ++i;

          if (i == component.size()) {
            // do last link then quit
            he = component.front();
            break;
          } else
            he = component[i];

        } while (!mesh.is_boundary(he));

        // next section: connect this section to the last if not already connected
        // and connect the previous links around vi instead
        HalfedgeId prev_he = mesh.reverse(component[i-1]);
        OTHER_ASSERT(mesh.is_boundary(prev_he));

        // if these were not next to each other in the loop around vi, make them
        if (mesh.prev(he) != prev_he) {

          // connect the boundary for vi after the last contiguous sections are removed
          mesh.unsafe_boundary_link(dangling, mesh.next(prev_he));

          // we have a new dangling halfedge
          dangling = mesh.prev(he);

          // connect the boundary for newv
          mesh.unsafe_boundary_link(prev_he, he);

          // We'll pass through this at least once (if the whole newv part is contiguous,
          // we'll pass here in the very end), so we can set an outgoing (boundary)
          // halfedge for newv and vi here. We'll do this when he is component[0].
          if (he == component.front()) {
            mesh.unsafe_set_halfedge(newv, he);
            mesh.unsafe_set_halfedge(vi, mesh.next(dangling));
          }
        }

      } while (he != component.front()); // stop once we're back at the start

      // no properties should have to change for this -- all faces/halfedges already exist,
      // the only change is the face -> vertex entries and some halfedge src vertices.
    }
  }

  // replace all intersection vertices with the first vertex on the list (in faces and boundaries)
  for (auto const &I : intersections) {
    if (I.type == itLoop)
      continue;

    for (auto vi : I.vertices()) {
      if (vi == I.vertices.front())
        continue;

      for (auto f : mesh.faces(vi)) {
        unsafe_replace_vertex(f, vi, I.vertices.front());
      }

      for (auto he : mesh.outgoing(vi)) {
        unsafe_set_src(he, I.vertices.front());
      }
    }
  }

  // merge all halfedges as indicated by edge_correspondence
  for (auto he : mesh.interior_halfedges()) {

    HalfedgeId re = edge_correspondences[he];

    // true internal edge
    if (!re.valid())
      continue;

    // already good (we will visit each edge twice)
    if (re == mesh.reverse(he))
      continue;

    OTHER_ASSERT(mesh.is_boundary(mesh.reverse(he)) && mesh.is_boundary(mesh.reverse(re)));

    // fix prev/next on adjacent boundaries to route around us
    if (mesh.prev(he) != re) {
      unsafe_boundary_link(mesh.prev(he), mesh.next(re));
    }
    if (mesh.next(he) != re) {
      unsafe_boundary_link(mesh.prev(re), mesh.next(he));
    }

    // delete attached boundary edges
    erase(mesh.reverse(he));
    erase(mesh.reverse(re));

    // set reverse
    unsafe_interior_link(he, re);
  }

  // find new boundary halfedges for intersection vertices that are still on the boundary
  for (auto const &I : intersections) {
    if (I.type == itLoop) {
      for (auto vi : I.vertices)
      mesh.ensure_boundary_halfedge(vi);
    } else
      mesh.ensure_boundary_halfedge(I.vertices.front());
  }
}

/*

= Peform Mesh CSG on a single mesh, extracting the contour at the given depth.

*/

void csg(MutableTriangleTopology &mesh, int target_depth, int pos_id_idx) {

  FieldId<TV, VertexId> pos_id(pos_id_idx);
  OTHER_ASSERT(mesh.has_field(pos_id));

  auto face_tree = mesh.face_tree(positions);
  auto edge_tree = mesh.edge_tree(positions);

  auto intersections = compute_edge_face_intersections(mesh, mesh.field(pos_id), face_tree, edge_tree);
  extend(intersections, compute_face_face_face_intersections(mesh, mesh.field(pos_id), face_tree));

  Tuple<Field<std::vector<int>, FaceId>,
        Field<std::vector<int>, HalfedgeId>,
        Field<int, VertexId>> path_info = assemble_paths(mesh, mesh.field(pos_id), intersections);

  // add the path fields to the mesh
  FieldId<std::vector<int>, FaceId> face_intersections_id = mesh.add_field(path_info.x);
  FieldId<std::vector<int>, HalfedgeId> edge_intersections_id = mesh.add_field(path_info.y);
  FieldId<int, VertexId> vertex_intersections_id = mesh.add_field(path_info.z);

  create_intersection_vertices(mesh, pos_id, intersections, vertex_intersections_id);

  Tuple<FieldId<HalfedgeId, HalfedgeId>,
        FieldId<FaceId, FaceId>> seam_info = retriangulate_faces(mesh, pos_id, intersections,
                                                                 face_intersections_id,
                                                                 edge_intersections_id,
                                                                 vertex_intersections_id);

  // use original mesh for raycasts, and shoot rays only at original vertices
  // to avoid having to deal with constructions
  Field<int, FaceId> depth = compute_depths(mesh, mesh.field(pos_id),
                                            mesh.field(vertex_intersections_id),
                                            mesh.field(seam_info.x), face_tree);

  // delete all faces that are not at our output depth
  for (auto fid : mesh.faces()) {
    if (depth[fid] != target_depth)
      mesh.erase(fid, true);
  }

  // stitch a mesh back together according to the given edge correspondence
  stitch_meshes(mesh, pos_id, intersections, vertex_intersections_id, seam_info.x);

  // remove all fields that are useless without the intersection data
  mesh.remove_field(face_intersections_id);
  mesh.remove_field(edge_intersections_id);
  mesh.remove_field(vertex_intersections_id);
  mesh.remove_field(seam_info.x);

  return seam_info.y;
}

}
