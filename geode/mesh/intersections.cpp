#include <geode/mesh/intersections.h>
#include <geode/utility/stl.h>
#include <geode/utility/prioritize.h>
#include <geode/geometry/SimplexTree.h>
#include <geode/geometry/traverse.h>
#include <geode/geometry/Ray.h>
#include <geode/math/fallback_sort.h>
#include <geode/vector/Rotation.h>
#include <geode/exact/scope.h>

// should be eliminated by move to all exact predicates, we need a new
// (likely non-delaunay) triangulator who can work without constructions
#include <geode/exact/delaunay.h>

namespace geode {

const bool debug_intersections = true;

/*

= Intersections

- compute all edge-face intersections, and store with affected face and
edge (ignore those that share a common vertex)

*/

typedef Vector<real,3> TV;

struct edge_face_intersection_helper {
  TriangleTopology const &mesh;
  Tuple<Ref<SimplexTree<TV,2>>, Array<FaceId>> face_tree;
  Tuple<Ref<SimplexTree<TV,1>>, Array<HalfedgeId>> edge_tree;

  std::vector<Intersection> intersections;

  edge_face_intersection_helper(TriangleTopology const &mesh,
                                Tuple<Ref<SimplexTree<TV,2>>, Array<FaceId>> face_tree,
                                Tuple<Ref<SimplexTree<TV,1>>, Array<HalfedgeId>> edge_tree)
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
      auto he = mesh.is_boundary(h) ? mesh.reverse(h) : h;
      Intersection intersection(he, f, ray.t_max, ray.point(ray.t_max));
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
                                                          Tuple<Ref<SimplexTree<TV,2>>, Array<FaceId>> face_tree,
                                                          Tuple<Ref<SimplexTree<TV,1>>, Array<HalfedgeId>> edge_tree) {
  edge_face_intersection_helper helper(mesh, face_tree, edge_tree);
  double_traverse(*face_tree.x, *edge_tree.x, helper);
  return helper.intersections;
}

/*

- compute all face-face-face intersections, and store with affected faces

*/

struct face_face_intersection_helper {
  TriangleTopology const &mesh;
  Tuple<Ref<SimplexTree<TV,2>>, Array<FaceId>> face_tree;

  // TODO EXACT: we store constructed edges, but they're only necessary for tree
  // creation/lookup, so giving the second traversal a thickness large enough to
  // contain the largest interval should be safe and easy.
  Array<Vector<int,2>> indices;
  Array<TV> vertices;
  Array<Tuple<FaceId, FaceId>> faces;

  face_face_intersection_helper(TriangleTopology const &mesh,
                                Tuple<Ref<SimplexTree<TV,2>>, Array<FaceId>> face_tree)
  : mesh(mesh), face_tree(face_tree) {}

  inline bool cull(int nf, int ne) const {
    return false;
  }
  inline bool cull(int n) const {
    return false;
  }

  void check_intersection(int nf1, int nf2) {
    // check if faces intersect, and add to generated edges if they do
    FaceId f1 = face_tree.y[nf1];
    FaceId f2 = face_tree.y[nf2];

    // don't let the same face intersect itself, and don't test pairs twice
    if (f1 >= f2)
      return;

    // don't check vertex-incident faces
    // even if there is a triple intersection of three faces, a least one pair 
    // of them must be non-loop. These two will then generate an edge that the 
    // third is tested against to reveal the intersection
    if (mesh.common_vertex(f1,f2).valid()) 
      return;

    Triangle<TV> const &face1 = face_tree.x->simplices[nf1];
    Triangle<TV> const &face2 = face_tree.x->simplices[nf2];
    Segment<TV> result;

    // TODO EXACT (not super-necessary to do exact if this is made conservative)
    if (face1.intersection(face2, result)) {
      indices.append(vec(vertices.size(), vertices.size()+1));
      vertices.append(result.x0);
      vertices.append(result.x1);
      faces.append(tuple(f1, f2));
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

  inline void leaf(int n) {
    leaf(n,n);
  }
};

struct face_face_face_intersection_helper {
  TriangleTopology const &mesh;

  Ref<SimplexTree<TV,1>> edge_tree;
  Array<Tuple<FaceId,FaceId>> const &edge_faces;

  Tuple<Ref<SimplexTree<TV,2>>, Array<FaceId>> face_tree;

  Hashtable<Vector<FaceId,3>> checked;

  std::vector<Intersection> intersections;

  face_face_face_intersection_helper(TriangleTopology const &mesh,
                                     Ref<SimplexTree<TV,1>> edge_tree,
                                     Array<Tuple<FaceId,FaceId>> const &edge_faces,
                                     Tuple<Ref<SimplexTree<TV,2>>, Array<FaceId>> face_tree)
  : mesh(mesh), edge_tree(edge_tree), edge_faces(edge_faces), face_tree(face_tree) {}

  inline bool cull(int nf, int ne) const {
    return false;
  }

  void check_intersection(int nf, int ne) {
    Vector<FaceId,3> f = vec(edge_faces[ne].x, edge_faces[ne].y, face_tree.y[nf]);

    // no edge face may be the third face (not caught by the common edge check)
    if (f.x == f.z || f.y == f.z)
      return;

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
    if (cv12.valid() && mesh.common_vertex(f.y,f.z) == cv12)
      return;

    // TODO EXACT using the three input faces
    Segment<TV> const &edge = edge_tree->simplices[ne];
    Ray<TV> ray(edge);
    Triangle<TV> const &face = face_tree.x->simplices[nf];

    if (face.intersection(ray)) {
      intersections.push_back(Intersection(f.x, f.y, f.z, ray.point(ray.t_max)));
    }
  }

  void leaf(int nf, int ne) {
    auto es = edge_tree->prims(ne);
    auto fs = face_tree.x->prims(nf);

    for (int ei : es) {
      for (int fi: fs) {
        check_intersection(fi, ei);
      }
    }
  }
};

std::vector<Intersection> compute_face_face_face_intersections(TriangleTopology const &mesh,
                                                               Tuple<Ref<SimplexTree<TV,2>>, Array<FaceId>> face_tree) {
  face_face_intersection_helper ff_helper(mesh, face_tree);
  double_traverse(*face_tree.x, ff_helper);
  auto edge_tree = new_<SimplexTree<TV,1>>(new_<SegmentSoup>(ff_helper.indices), ff_helper.vertices, 1);
  face_face_face_intersection_helper fff_helper(mesh, edge_tree, ff_helper.faces, face_tree);
  double_traverse(*face_tree.x, *edge_tree, fff_helper);
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

// Use this to look up edge intersections (takes care of looking up by lower index halfedge)
struct GetEdgeIntersections {
  TriangleTopology const &mesh;
  Field<int, HalfedgeId> const &edge_intersections;
  std::vector<std::vector<int>> &intersection_indices;

  GetEdgeIntersections(TriangleTopology const &mesh,
                       Field<int, HalfedgeId> const &ei,
                       std::vector<std::vector<int>> &intersection_indices)
  : mesh(mesh), edge_intersections(ei), intersection_indices(intersection_indices) {};

  std::vector<int> &operator[](HalfedgeId id) {
    HalfedgeId r = mesh.reverse(id);
    if (!mesh.is_boundary(r) && r < id) {
      GEODE_ASSERT(edge_intersections[id] == -1 || intersection_indices[edge_intersections[id]].empty());
      id = r;
    } else {
      GEODE_ASSERT(edge_intersections[r] == -1 || intersection_indices[edge_intersections[r]].empty());
    }

    GEODE_ASSERT(edge_intersections[id] != -1);
    return intersection_indices[edge_intersections[id]];
  }

  std::vector<int> get_ordered(HalfedgeId id) {

    HalfedgeId r = mesh.reverse(id);
    bool flip = false;
    if (!mesh.is_boundary(r) && r < id) {
      GEODE_ASSERT(edge_intersections[id] == -1 || intersection_indices[edge_intersections[id]].empty());
      flip = true;
      id = r;
    } else {
      GEODE_ASSERT(edge_intersections[r] == -1 || intersection_indices[edge_intersections[r]].empty());
    }

    if (edge_intersections[id] == -1)
      return std::vector<int>();

    std::vector<int> ints = intersection_indices[edge_intersections[id]];

    // sort them for our orientation
    if (flip)
      std::reverse(ints.begin(), ints.end());

    return ints;
  }
};

Tuple<Field<int, FaceId>,
      Field<int, HalfedgeId>,
      Field<int, VertexId>,
      std::vector<std::vector<int>>> assemble_paths(TriangleTopology const &mesh,
                                                    Field<TV, VertexId> const &pos,
                                                    std::vector<Intersection> &intersections) {

  // make fields compatible with the mesh
  int id = 0;
  Field<int, FaceId> face_intersections = mesh.create_compatible_face_field<int>();
  for (auto &f : face_intersections.flat) {
    f = id++;
  }
  Field<int, HalfedgeId> edge_intersections = mesh.create_compatible_halfedge_field<int>();
  for (auto &f : edge_intersections.flat) {
    f = id++;
  }
  Field<int, VertexId> vertex_intersections = mesh.create_compatible_vertex_field<int>();
  for (auto &f : vertex_intersections.flat) {
    f = -1;
  }

  std::vector<std::vector<int>> intersection_indices(face_intersections.flat.size() + edge_intersections.flat.size());

  // register regular intersections in fields
  for (int i = 0; i < (int)intersections.size(); ++i) {
    Intersection const &I = intersections[i];

    switch (I.type) {
      case Intersection::itEF: {
        intersection_indices[face_intersections[I.data.ef.face]].push_back(i);
        intersection_indices[edge_intersections[I.data.ef.edge]].push_back(i); // this is always the internal halfedge with smallest index
        break;
      }
      case Intersection::itFFF:
        break;
      case Intersection::itLoop:
      default:
        GEODE_UNREACHABLE();
    }
  }

  GetEdgeIntersections get_edge_intersections(mesh, edge_intersections, intersection_indices);

  // a map from face pairs to fff intersections
  Hashtable<Tuple<FaceId, FaceId>, std::vector<Tuple<int, FaceId>>> fff_by_edge;

  // walk paths and fill in connected_to, creating loop vertices as we go
  for (int i = 0; i < (int)intersections.size(); ++i) {
    if (intersections[i].type == Intersection::itEF) {
      std::cout << "Connections for " << i << ": " << intersections[i] << std::endl;

      auto f = intersections[i].data.ef.face;
      auto e = intersections[i].data.ef.edge;
      // for all incident faces to the edge, find the next intersection and connect them
      for (auto fi : mesh.faces(e)) {
        // the edge is a boundary edge, nothing to do here
        if (!fi.valid())
          continue;

        bool found = false;

        // fi intersects one of the edges of f (debug only: check that it's exactly one)
        for (auto ei : mesh.halfedges(f)) {
          for (int i2 : get_edge_intersections[ei]) {
            Intersection const &I2 = intersections[i2];
            GEODE_ASSERT(I2.type == Intersection::itEF);

            if (I2.data.ef.face == fi) {
              std::cout << "  connected to " << i2 << ": " << I2 << " (case 1)" << std::endl;
              GEODE_ASSERT(!found); // fi cannot intersect more than one edge of f since one of its edges intersects f
              found = true;
              intersections[i].connected_to.append(i2);
              if (!debug_intersections)
                break;
            }
          }

          if (!debug_intersections && found)
            break;
        }

        if (!debug_intersections && found)
          continue;

        // another edge ei != e of fi intersects f (debug only: check that it's exactly one)
        for (auto ei : mesh.halfedges(fi)) {
          if (ei == e || ei == mesh.reverse(e))
            continue;
          for (int i2 : get_edge_intersections[ei]) {
            Intersection const &I2 = intersections[i2];
            GEODE_ASSERT(I2.type == Intersection::itEF);
            if (I2.data.ef.face == f) {
              std::cout << "  connected to " << i2 << ": " << I2 << " (case 2)" << std::endl;
              GEODE_ASSERT(!found); // only a maximum of two edges of fi can intersect f
              found = true;
              intersections[i].connected_to.append(i2);
              if (!debug_intersections)
                break;
            }
          }

          if (!debug_intersections && found)
            break;
        }

        if (!debug_intersections && found)
          continue;

        // fi and f share a loop vertex
        auto lv = mesh.common_vertex(fi, f);
        if (lv.valid()) {
          std::cout << "  connected to loop vertex" << std::endl;
          GEODE_ASSERT(!found);

          if (vertex_intersections[lv] == -1) {
            // add an intersection for the loop vertex
            vertex_intersections[lv] = intersections.size();
            intersections.push_back(Intersection(lv, pos[lv]));
          }

          intersections[i].connected_to.append(vertex_intersections[lv]);
          intersections[vertex_intersections[lv]].connected_to.append(i);

          continue;
        }

        if (debug_intersections) {
          GEODE_ASSERT(found);
          continue;
        }

        // otherwise, this must be an error.
        GEODE_ASSERT(false);
      }
    } else if (intersections[i].type == Intersection::itFFF) {
      Intersection const &I = intersections[i];
      // make a map from face pairs to face-face-face intersections
      Tuple<FaceId, FaceId> p1(I.data.fff.face1, I.data.fff.face2);
      Tuple<FaceId, FaceId> p2(I.data.fff.face1, I.data.fff.face3);
      Tuple<FaceId, FaceId> p3(I.data.fff.face2, I.data.fff.face3);

      fff_by_edge[p1].push_back(tuple(i, I.data.fff.face3));
      fff_by_edge[p2].push_back(tuple(i, I.data.fff.face2));
      fff_by_edge[p3].push_back(tuple(i, I.data.fff.face1));
    } else if (intersections[i].type == Intersection::itLoop) {
      // all loop vertices are at the end, they should already be connected, quit the loop
      break;
    } else {
      // there should not be anything else
      std::cout << "found illegal intersection: " << intersections[i] << std::endl;
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
    for (HalfedgeId ei : mesh.halfedges(fp.x)) {
      for (int eii : get_edge_intersections[ei]) {
        GEODE_ASSERT(intersections[eii].type == Intersection::itEF);
        if (intersections[eii].data.ef.face == fp.y) {
          if (end1 == -1) {
            end1 = eii;
          } else {
            GEODE_ASSERT(end2 == -1);
            end2 = eii;
            if (!debug_intersections)
              goto found_all;
          }
        }
      }
    }

    // does any edge of fp.y have an intersection with fp.x?
    for (HalfedgeId ei : mesh.halfedges(fp.y)) {
      for (int eii : get_edge_intersections[ei]) {
        GEODE_ASSERT(intersections[eii].type == Intersection::itEF);
        if (intersections[eii].data.ef.face == fp.x) {
          if (end1 == -1) {
            end1 = eii;
          } else {
            GEODE_ASSERT(end2 == -1);
            end2 = eii;
            if (!debug_intersections)
              goto found_all;
          }
        }
      }
    }

    // now, we ought to have collected exactly two intersections.
    GEODE_ASSERT(end1 != -1 && end2 != -1);

    found_all:

    // remove the path edge between the end points
    GEODE_ASSERT(intersections[end1].connected_to.contains(end2));
    GEODE_ASSERT(intersections[end2].connected_to.contains(end1));

    intersections[end1].connected_to.remove_first_lazy(end2);
    intersections[end2].connected_to.remove_first_lazy(end1);

    std::cout << "inserting fff intersections between " << end1 << " and " << end2 << ": " << ints << std::endl; 

    if (debug_intersections || ints.size() > 1) {
      IntervalScope S;

      // TODO EXACT
      // compute the edge implied by the face pair
      TV N = cross(mesh.triangle(fp.x, pos).normal(),
                   mesh.triangle(fp.y, pos).normal());

      std::vector<Prioritize<int>> sorted_ints;
      for (auto I : ints) {
        real d = dot(Vector<Interval,3>(N), intersections[I.x].construct).center();
        sorted_ints.push_back(prioritize(I.x, d));
      }

      // add the end points
      real d1 = dot(Vector<Interval,3>(N), intersections[end1].construct).center();
      real d2 = dot(Vector<Interval,3>(N), intersections[end2].construct).center();
      sorted_ints.push_back(prioritize(end1, d1));
      sorted_ints.push_back(prioritize(end2, d2));

      // sort all intersections (smallest first)
      fallback_sort(sorted_ints, std::less<Prioritize<int>>());

      // make sure the end points are in fact at the ends
      GEODE_ASSERT(sorted_ints.front().a == end1 || sorted_ints.front().a == end2);
      GEODE_ASSERT(sorted_ints.back().a == end1 || sorted_ints.back().a == end2);

      std::cout << "  sorted: " << sorted_ints << std::endl;

      // insert the sorted string of face-face-face intersections into the path
      // between the end points
      for (int ii = 0; ii < (int)sorted_ints.size()-1; ++ii) {
        int i1 = sorted_ints[ii].a;
        int i2 = sorted_ints[ii+1].a;

        GEODE_ASSERT(!intersections[i1].connected_to.contains(i2));
        GEODE_ASSERT(!intersections[i2].connected_to.contains(i1));
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
  for (int i = 0; i < (int)intersections.size(); ++i) {
    Intersection const &I = intersections[i];

    switch (I.type) {
      case Intersection::itEF:
        break;
      case Intersection::itFFF: {
        intersection_indices[face_intersections[I.data.fff.face1]].push_back(i);
        intersection_indices[face_intersections[I.data.fff.face2]].push_back(i);
        intersection_indices[face_intersections[I.data.fff.face3]].push_back(i);
        break;
      }
      case Intersection::itLoop:
        // we can quit, once we're in loop territory that's all we'll see
        i = intersections.size();
        break;
      default:
        GEODE_UNREACHABLE();
    }
  }

  // debug only: make sure all connections are symmetric,  intersections have
  // the correct number of connections, and everything is registered in the mesh 
  if (debug_intersections) {
    for (int i = 0; i < (int)intersections.size(); ++i) {
      Intersection const &I = intersections[i];

      GEODE_ASSERT(I.connected_to.is_unique());

      switch (I.type) {
        case Intersection::itEF: {
          GEODE_ASSERT(mesh.is_boundary(I.data.ef.edge) && I.connected_to.size() == 1 ||
                       !mesh.is_boundary(I.data.ef.edge) && I.connected_to.size() == 2);
          break;
        }
        case Intersection::itFFF: {
          GEODE_ASSERT(I.connected_to.size() == 6);
          break;
        }
        case Intersection::itLoop: {
          GEODE_ASSERT(I.connected_to.size() % 2 == 0 && !I.connected_to.empty());
          break;
        }
      }

      for (int i2 : I.connected_to) {
        Intersection const &I2 = intersections[i2];
        GEODE_ASSERT(I2.connected_to.contains(i));
      }
    }
  }

  return tuple(face_intersections, edge_intersections, vertex_intersections, intersection_indices);
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

void create_intersection_vertices(MutableTriangleTopology &mesh, FieldId<TV, VertexId> pos_id,
                                  std::vector<Intersection> &intersections,
                                  FieldId<int,VertexId> vertex_intersections_id) {
  auto &pos = mesh.field(pos_id);
  auto &vertex_intersections = mesh.field(vertex_intersections_id);

  for (int i = 0; i < (int)intersections.size(); ++i) {
    Intersection &I = intersections[i];
    GEODE_ASSERT(I.vertices.empty());
    switch (I.type) {
      case Intersection::itEF:
        I.vertices.extend(vec(mesh.add_vertex(), mesh.add_vertex()));
        break;
      case Intersection::itFFF:
        I.vertices.extend(vec(mesh.add_vertex(), mesh.add_vertex(), mesh.add_vertex()));
        break;
      case Intersection::itLoop:
        // we may have to duplicate loop vertices, but not for triangulation,
        // this will only become necessary during stitching, so we'll do that later
        break;
      default:
        GEODE_UNREACHABLE();
    }

    for (auto vi : I.vertices) {
      vertex_intersections[vi] = i;
      pos[vi] = vec(I.construct.x.center(), I.construct.y.center(), I.construct.z.center());
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

template<class TA, class Index=int>
struct indirect_less {
  TA const &ta;

  indirect_less(TA const &ta): ta(ta) {
  }

  inline bool operator()(Index i1, Index i2) const {
    return ta[i1] < ta[i2];
  }
};

// a version of mesh.reverse that works on deleted faces
HalfedgeId former_reverse(TriangleTopology const &mesh, HalfedgeId id) {
  // only for internal halfedges
  assert(id.id >= 0);
  const int f = id.id/3;
  return mesh.faces_.flat[f].neighbors[id.id-3*f];
}

Tuple<FieldId<HalfedgeId, HalfedgeId>,
      FieldId<FaceId, FaceId>> retriangulate_faces(MutableTriangleTopology &mesh,
                                                   FieldId<TV, VertexId> pos_id, 
                                                   std::vector<Intersection> const &intersections,
                                                   std::vector<std::vector<int>> &intersection_indices,
                                                   FieldId<int, FaceId> face_intersections_id,
                                                   FieldId<int, HalfedgeId> edge_intersections_id,
                                                   FieldId<int, VertexId> vertex_intersections_id) {

  // add result fields (before getting references to the existing fields)
  auto edge_correspondences_id = mesh.add_halfedge_field<HalfedgeId>();
  auto face_correspondences_id = mesh.add_face_field<FaceId>();
  auto &edge_correspondences = mesh.field(edge_correspondences_id);
  auto &face_correspondences = mesh.field(face_correspondences_id);

  // initialize face correspondences
  face_correspondences.flat.fill(FaceId());

  auto &pos = mesh.field(pos_id);
  auto &vertex_intersections = mesh.field(vertex_intersections_id);
  auto &edge_intersections = mesh.field(edge_intersections_id);
  auto &face_intersections = mesh.field(face_intersections_id);

  // sort all edge intersections for t value along their edges
  // TODO EXACT
  for (auto he : mesh.interior_halfedges()) {
    fallback_sort(intersection_indices[edge_intersections[he]], indirect_less<std::vector<Intersection>>(intersections));
  }

  // store properly sorted (reversed) edge intersection on the other halfedges
  // we only copy from lower to higher edges (because that's how it's stored)
  for (auto he : mesh.interior_halfedges()) {
    auto r = mesh.reverse(he);
    if (mesh.is_boundary(r))
      continue;

    // opposite edge has data but we don't? Reverse and copy it.
    auto &r_int = intersection_indices[edge_intersections[r]];
    auto &he_int = intersection_indices[edge_intersections[he]];
    if (!r_int.empty() && he_int.empty()) {
      GEODE_ASSERT(r < he);
      he_int.resize(r_int.size());
      std::copy(r_int.rbegin(), r_int.rend(), he_int.begin());
    }
  }

  // Map from a sorted pair of intersections to pairs of vertices. Each pair of vertices
  // is sorted the same (vertex corresponding to lower index intersection first).
  // There should be zero or two vertex pairs for any given intersection pair.
  Hashtable<Vector<int,2>, Array<Vector<VertexId,2>>> intersection_edges;

  auto faces = mesh.faces();
  for (auto const f : faces) {

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
    //std::cout << "intersections on face " << f << std::endl;

    // add boundary vertices and edges
    for (int i = 0; i < 3; ++i) {
      HalfedgeId he = mesh.halfedge(f, i);
      GEODE_DEBUG_ONLY(VertexId startv = mesh.vertex(f, i));
      assert(mesh.src(he) == startv);
      assert(startv == vertices[i]);

      // check where the intersections are stored (we're not iterating into new faces with new edges)
      GEODE_ASSERT(edge_intersections[he] != -1);
      std::vector<int> ints = intersection_indices[edge_intersections[he]];

      /*
      if (mesh.is_boundary(mesh.reverse(he)))
        std::cout << "  he " << he << "(" << mesh.src(he) << "-" << mesh.dst(he) << "): " << ints << " (" << intersection_indices[edge_intersections[he]] << "/<boundary>)" << std::endl;
      else if (edge_intersections[mesh.reverse(he)] == -1) 
        std::cout << "  he " << he << "(" << mesh.src(he) << "-" << mesh.dst(he) << "): " << ints << " (" << intersection_indices[edge_intersections[he]] << "/<new>)" << std::endl;
      else
        std::cout << "  he " << he << "(" << mesh.src(he) << "-" << mesh.dst(he) << "): " << ints << " (" << intersection_indices[edge_intersections[he]] << "/" << intersection_indices[edge_intersections[mesh.reverse(he)]] << ")" << std::endl;
      */

      // add all vertices contained in these constraints
      for (auto ii : ints) {
        // all these intersections intersect our edge
        GEODE_ASSERT(intersections[ii].type == Intersection::itEF);
        // make sure that the halfedge x that this face intersected is he,
        // or that x used to be the reverse of he (can't use reverse(), because
        // the face containing x has already been deleted)
        GEODE_ASSERT(intersections[ii].data.ef.edge == he || former_reverse(mesh, intersections[ii].data.ef.edge) == he);
        intersection_to_local_idx[ii] = vertices.size();
        // first vertex is the one dedicated to the edge
        vertices.append(intersections[ii].vertices[0]);
      }

      // add triangulation constraints along this edge (none of these are intersection edges)
      if (!ints.empty()) {
        edges.append(vec(i, intersection_to_local_idx[ints.front()]));
        for (int j = 1; j < (int)ints.size(); ++j) {
          edges.append(vec(intersection_to_local_idx[ints[j-1]],
                              intersection_to_local_idx[ints[j]]));
        }
        edges.append(vec(intersection_to_local_idx[ints.back()], (i+1)%3));
      } else {
        edges.append(vec(i, (i+1)%3));
      }

      // make all edge to edge triangulation constraints (using connected_to) 
      for (int ii : ints) {
        Intersection const &I = intersections[ii];
        for (int i2 : I.connected_to) {
          if (!intersection_to_local_idx.contains(i2))
            continue; // we will add this edge when we add i2's vertex 
                      // (or i2 is not on the boundary of this face, and we'll never add it)

          // at this point, only EF vertices on the boundaries of f are in intersection_to_local_idx
          // we can only connect to other EF vertices on another edge of f
          GEODE_ASSERT(intersections[i2].type == Intersection::itEF);
          GEODE_ASSERT(intersections[i2].data.ef.face == I.data.ef.face);
          GEODE_ASSERT(intersections[i2].data.ef.edge != he && intersections[i2].data.ef.edge != mesh.reverse(he));
          GEODE_ASSERT(mesh.halfedges(f).contains(intersections[i2].data.ef.edge) || 
                       mesh.halfedges(f).contains(former_reverse(mesh, intersections[i2].data.ef.edge)));

          edges.append(vec(intersection_to_local_idx[ii], intersection_to_local_idx[i2]));

          //std::cout << "ee intersection edge " << ii << "-" << i2 << ", vertices " << vec(vertices[intersection_to_local_idx[ii]],
          //                                                                            vertices[intersection_to_local_idx[i2]]) << std::endl;
          //std::cout << "  i1: " << intersections[ii] << std::endl;
          //std::cout << "  i2: " << intersections[i2] << std::endl;

          // Remember vertex pairs of intersection edges for later reconstruction of
          // edge correspondence
          if (ii < i2) {
            intersection_edges[vec(ii,i2)].append(vec(vertices[intersection_to_local_idx[ii]],
                                                     vertices[intersection_to_local_idx[i2]]));
          } else {
            intersection_edges[vec(i2,ii)].append(vec(vertices[intersection_to_local_idx[i2]],
                                                     vertices[intersection_to_local_idx[ii]]));
          }
        }
      }
    }

    // make all interior vertices and interior edge constraints (using
    // connected_to).

    for (int i : intersection_indices[face_intersections[f]]) {
      Intersection const &I = intersections[i];
      intersection_to_local_idx[i] = vertices.size();

      // get the vertex for this intersection and add it to the set of vertices
      // to triangulate
      int vidx = -1;
      switch (I.type) {
        case Intersection::itEF:
          // we must be the face in this intersection
          GEODE_ASSERT(I.data.ef.face == f);
          // first vertex is the one dedicated to the edge, second one to the face
          vidx = 1;
          break;
        case Intersection::itFFF:
          if (I.data.fff.face1 == f) {
            vidx = 0;
          } else if (I.data.fff.face2 == f) {
            vidx = 1;
          } else if (I.data.fff.face3 == f) {
            vidx = 2;
          } else {
            GEODE_FATAL_ERROR();
          }
          break;
        case Intersection::itLoop:
        default:
          GEODE_UNREACHABLE();
      }
      vertices.append(I.vertices[vidx]);

      // go through all connected intersections, and remember edges to vertices
      // that are already there (loop vertices are already there).
      for (int i2 : I.connected_to) {
        if (!intersection_to_local_idx.contains(i2))
          continue; // we will add this edge when we add i2's vertex

        edges.append(vec(intersection_to_local_idx[i], intersection_to_local_idx[i2]));

        //std::cout << "internal intersection edge " << i << "-" << i2 << ", vertices " << vec(vertices[intersection_to_local_idx[i]],
        //                                                                            vertices[intersection_to_local_idx[i2]]) << std::endl;
        //std::cout << "  i1: " << intersections[i] << std::endl;
        //std::cout << "  i2: " << intersections[i2] << std::endl;

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

    // only replace faces that have intersections on them
    if (vertices.size() == 3) {
      continue;
    }

    // TODO EXACT: this should be replaced by an exact non-delaunay version of
    // recursive triangulation
    // project points into the face plane so delaunay can deal with them
    Array<Vector<real,2>> projected;
    // make a Frame such that applying the frame rotates the face normal to z
    auto rot = Rotation<TV>::from_rotated_vector(mesh.normal(f, pos), vec(0., 0., 1.));
    for (auto v : vertices) {
      projected.append((rot*pos[v]).xy());
    }
    Ref<MutableTriangleTopology> faces = new_<MutableTriangleTopology>(*delaunay_points(projected, edges));

    // erode all faces that are not protected by constraint edges 
    bool deleting;
    do {
      deleting = false;
      for (HalfedgeId b : faces->boundary_edges()) {
        Vector<int,2> e(faces->src(b).idx(), faces->dst(b).idx());
        if (!edges.contains(e) && !edges.contains(vec(e.y,e.x))) {
          deleting = true;
          faces->erase(faces->face(faces->reverse(b)));
        }
      }
    } while (deleting);

    // replace original face with new faces
    //std::cout << "  adding " << faces->n_faces() << " new faces, erasing face " << f << ": " << mesh.vertices(f) << std::endl;
    mesh.erase(f, false);
    for (auto newf : faces->faces()) {
      Vector<VertexId,3> verts = faces->vertices(newf);
      Vector<VertexId,3> mverts = vec(vertices[verts[0].id],
                                      vertices[verts[1].id], 
                                      vertices[verts[2].id]);
      //std::cout << "  adding face " << verts << " -> " << mverts << std::endl;
      auto added = mesh.add_face(mverts);

      // fill face_correspondences
      face_correspondences[added] = f;

      // make sure edge_intersections and face_intersections for the new face 
      // points nowhere
      for (HalfedgeId id : mesh.halfedges(added)) {
        edge_intersections[id] = -1;
      }
      face_intersections[added] = -1;
    }
  }

  // make sure all intersection edges have a valid number of vertex representatives, 
  // and set edge correspondences
  edge_correspondences.flat.fill(HalfedgeId());
  for (auto p : intersection_edges) {
    GEODE_ASSERT(p.data().size() == 2);

    GEODE_ASSERT(intersections[p.key().x].vertices.contains(p.data()[0][0]));
    GEODE_ASSERT(intersections[p.key().y].vertices.contains(p.data()[0][1]));
    GEODE_ASSERT(intersections[p.key().x].vertices.contains(p.data()[1][0]));
    GEODE_ASSERT(intersections[p.key().y].vertices.contains(p.data()[1][1]));

    // get the edges corresponding to the two vertex pairs
    HalfedgeId he1 = mesh.halfedge(p.data().front().x, p.data().front().y);
    HalfedgeId he2 = mesh.halfedge(p.data().back().x, p.data().back().y);

    // the two halfedges are aligned, add four edge_correspondences entries
    edge_correspondences[he1] = mesh.reverse(he2);
    edge_correspondences[he2] = mesh.reverse(he1);
    edge_correspondences[mesh.reverse(he1)] = he2;
    edge_correspondences[mesh.reverse(he2)] = he1;

    //std::cout << "  adding edge correspondences: " << he1 << " <-> " << mesh.reverse(he2) << ", " << he2 << " <-> " << mesh.reverse(he1) << std::endl;
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
                                  Tuple<Ref<SimplexTree<TV,2>>, Array<FaceId>> const &face_tree) {

  Field<int, FaceId> depth_map = mesh.create_compatible_face_field<int>();
  Field<bool, FaceId> depth_assigned = mesh.create_compatible_face_field<bool>();
  Field<bool, VertexId> v_depth_assigned = mesh.create_compatible_vertex_field<bool>();

  for (auto vi : mesh.vertices()) {
    // find an original vertex (vertex_intersections[vi] == -1) which isn't assigned
    if (v_depth_assigned[vi] || vertex_intersections[vi] != -1)
      continue;

    // compute the depth of this vertex
    // TODO EXACT
    // this simple normal is not safe -- the normal must point "outside"
    Ray<TV> ray(pos[vi], mesh.normal(vi, pos), true);
    Array<Ray<TV>> intersections = face_tree.x->intersections(ray, 0);

    int depth = 0;
    for (auto r : intersections) {
      auto f = face_tree.y[r.aggregate_id];

      if (mesh.vertices(f).contains(vi))
        continue;

      auto simplex = face_tree.x->simplices[r.aggregate_id];
      if (dot(r.direction, simplex.normal()) > 0) {
        depth++;
      } else {
        depth--;
      }
    }

    // propagate the depth across this connected component
    std::vector<FaceId> upnext;

    // initialize any face incident to vi (be wary of boundaries)
    FaceId fi = mesh.face(mesh.halfedge(vi));
    if (!fi.valid())
      fi = mesh.face(mesh.reverse(mesh.halfedge(vi)));
    GEODE_ASSERT(fi.valid());
    GEODE_ASSERT(!depth_assigned[fi]);
    depth_assigned[fi] = true;
    depth_map[fi] = depth;
    for (auto vi : mesh.vertices(fi)) {
      v_depth_assigned[vi] = true;
    }

    //std::cout << "seeding depth " << depth << " at face " << fi << ", vertices " << mesh.vertices(fi) << std::endl;

    upnext.push_back(fi);

    while (!upnext.empty()) {
      FaceId id = upnext.back();
      depth = depth_map[id];
      upnext.pop_back();

      for (auto he : mesh.halfedges(id)) {
        HalfedgeId intersection_edge = edge_correspondences[he];

        #define SET_DEPTH(halfedge, depth) { \
          GEODE_ASSERT(!mesh.erased(halfedge)); \
          FaceId f = mesh.face(halfedge); \
          if (depth_assigned[f]) \
            GEODE_ASSERT((depth) == depth_map[f]); \
          else { \
            /*std::cout << "  setting depth " << depth << " at face " << f << ", vertices " << mesh.vertices(f) << std::endl;*/\
            GEODE_ASSERT(v_depth_assigned[mesh.src(halfedge)]); \
            GEODE_ASSERT(v_depth_assigned[mesh.dst(halfedge)]); \
            v_depth_assigned[mesh.opposite(halfedge)] = true; \
            depth_assigned[f] = true; \
            depth_map[f] = (depth); \
            upnext.push_back(f); \
          } \
        }

        if (intersection_edge.valid()) {
          // propagate to next bit of the same surface
          v_depth_assigned[mesh.src(intersection_edge)] = true;
          v_depth_assigned[mesh.dst(intersection_edge)] = true;
          SET_DEPTH(intersection_edge, depth);

          // is the other surface deeper or less deep?
          // TODO EXACT
          int diff = mesh.triangle(mesh.face(intersection_edge), pos).phi(pos[mesh.opposite(he)]) > 0 ? 1 : -1;

          // propagate to other surface 
          HalfedgeId r = mesh.reverse(he);
          HalfedgeId rie = edge_correspondences[r];
          GEODE_ASSERT(rie.valid());
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
                   std::vector<Intersection> &intersections,
                   FieldId<int, VertexId> vertex_intersections_id,
                   FieldId<HalfedgeId, HalfedgeId> edge_correspondences_id) {

  auto pos = mesh.field(pos_id);
  auto vertex_intersections = mesh.field(vertex_intersections_id);
  auto edge_correspondences = mesh.field(edge_correspondences_id);

  // loop vertices may have to be split, they may be attached to several connected
  // components in the new mesh. Count the number of components around the loop
  // vertex first, split the vertex and redistribute the incident faces before
  // merging intersection edges.
  auto orig_vertices = mesh.vertices();
  for (auto vi : orig_vertices) {
    int iid = vertex_intersections[vi];
    if (iid == -1)
      continue;
    if (intersections[iid].type != Intersection::itLoop)
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
      GEODE_ASSERT(mesh.is_boundary(component.front()));

      // walk right (in the direction of where the faces connected to our halfedge
      // are), and add faces to our component until we have closed the loop
      while (true) {
        HalfedgeId ohe = mesh.reverse(he);
        if (mesh.face(ohe).valid()) {
          // next one is connected via a face
          HalfedgeId next_he = mesh.next(ohe);
          // can't finish with a face, since the first he was a boundary
          GEODE_ASSERT(next_he != component[0]);
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
            GEODE_ASSERT(mesh.is_boundary(next_he));
            GEODE_ASSERT(vertex_intersections[mesh.dst(he)] == vertex_intersections[mesh.dst(next_he)]);
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

      // check if this component is all the outgoing edges there are (then we're good here)
      unsigned int n_outgoing = 0;
      for (auto oh: mesh.outgoing(vi)) {
        GEODE_ASSERT(mesh.src(oh) == vi); // stupid, but avoid unused variable warning
        n_outgoing++;
      }

      if (component.size() == n_outgoing) {
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
      he = component.front();

      // store the halfedge that is left dangling when we take this section out
      // of the loop
      HalfedgeId dangling = mesh.prev(he);

      do {

        // he is always a boundary halfedge here
        GEODE_ASSERT(mesh.is_boundary(he));

        // set the src vertex on the outgoing boundary edges
        mesh.unsafe_set_src(he, newv);

        // move on to the next (which cannot be past the end, and cannot be a boundary)
        ++i;
        GEODE_ASSERT(i < (int)component.size());
        he = component[i];
        GEODE_ASSERT(!mesh.is_boundary(he));

        // replace vi with newv in all faces in this section
        do {
          bool changed = mesh.unsafe_replace_vertex(mesh.face(he), vi, newv);
          GEODE_ASSERT(changed);

          ++i;

          if (i == (int)component.size()) {
            // do last link then quit
            he = component.front();
            break;
          } else
            he = component[i];

        } while (!mesh.is_boundary(he));

        // next section: connect this section to the last if not already connected
        // and connect the previous links around vi instead
        HalfedgeId prev_he = mesh.reverse(component[i-1]);
        GEODE_ASSERT(mesh.is_boundary(prev_he));

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

  // replace all intersection vertices with the first non-erased vertex on the list 
  for (auto const &I : intersections) {
    if (I.type == Intersection::itLoop)
      continue;

    std::cout << "merging vertices for " << I << std::endl;

    VertexId vnew;
    for (auto &vi : I.vertices) {
      std::cout << "  vertex " << vi;
      if (mesh.erased(vi)) {
        std::cout << " (erased)" << std::endl;
      } else {
        if (!vnew.valid()) {
          vnew = vi;
          // make sure newv is first in I.vertices
          vi = I.vertices[0];
          I.vertices[0] = vnew;
          if (!debug_intersections)
            break;
          std::cout << " (target)";
        }
        std::cout << " incident faces " << mesh.incident_faces(vi) << std::endl;
      }
    }

    if (!vnew.valid())
      continue;

    for (auto vi : I.vertices) {
      if (vi == vnew || mesh.erased(vi))
        continue;

      std::cout << "  replacing " << vi << " with " << vnew << std::endl;

      // replace vertex in faces
      for (auto f : mesh.incident_faces(vi)) {
        std::cout << "    replacing in face " << f << ": " << mesh.vertices(f) << std::endl;
        mesh.unsafe_replace_vertex(f, vi, vnew);
      }
      // replace vertex in boundaries
      for (auto he : mesh.outgoing(vi)) {
        if (mesh.is_boundary(he)) {
          std::cout << "    replacing src vertex in halfedge " << he << std::endl;
          mesh.unsafe_set_src(he, vnew);
        }
      }
      // mark vertex as deleted
      mesh.unsafe_set_erased(vi);
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

    HalfedgeId rhe = mesh.reverse(he);
    HalfedgeId rre = mesh.reverse(re);
    GEODE_ASSERT(mesh.valid(rhe) && mesh.valid(rre));
    GEODE_ASSERT(mesh.is_boundary(rhe) && mesh.is_boundary(rre));

    // Make sure that if their src vertex use them as their halfedge, we find 
    // another edge for the vertex to use (This temporarily break is_boundary 
    // on vertices. We take care of ensuring boundariness later).
    if (mesh.halfedge(mesh.src(rhe)) == rhe)
      mesh.unsafe_set_halfedge(mesh.src(rhe), mesh.left(rhe));
    if (mesh.halfedge(mesh.src(rre)) == rre)
      mesh.unsafe_set_halfedge(mesh.src(rre), mesh.left(rre));

    // fix prev/next on adjacent boundaries to route around us
    if (mesh.prev(rhe) != rre)
      mesh.unsafe_boundary_link(mesh.prev(rhe), mesh.next(rre));
    if (mesh.next(rhe) != rre) 
      mesh.unsafe_boundary_link(mesh.prev(rre), mesh.next(rhe));

    // delete attached boundary edges. 
    mesh.unsafe_set_erased(rhe);
    mesh.unsafe_set_erased(rre);

    // set reverse
    mesh.unsafe_interior_link(he, re);
  }

  // find new boundary halfedges for intersection vertices that are still on the boundary
  for (auto const &I : intersections) {
    if (I.type == Intersection::itLoop) {
      for (auto vi : I.vertices) {
        if (!mesh.erased(vi))
          mesh.ensure_boundary_halfedge(vi);
      }
    } else {
      // make sure the remaining vertex has a valid halfedge
      auto vi = I.vertices.front();
      if (!mesh.erased(vi))
        mesh.ensure_boundary_halfedge(vi);
    }
  }
}

/*

= Peform Mesh CSG on a single mesh, extracting the contour at the given depth.

*/

FieldId<FaceId, FaceId> csg(MutableTriangleTopology &mesh, int target_depth, int pos_id_idx) {

  FieldId<TV, VertexId> pos_id(pos_id_idx);
  GEODE_ASSERT(mesh.has_field(pos_id));

  auto face_tree = mesh.face_tree(mesh.field(pos_id));
  auto edge_tree = mesh.edge_tree(mesh.field(pos_id));

  auto intersections = compute_edge_face_intersections(mesh, face_tree, edge_tree);
  extend(intersections, compute_face_face_face_intersections(mesh, face_tree));

  Tuple<Field<int, FaceId>,
        Field<int, HalfedgeId>,
        Field<int, VertexId>,
        std::vector<std::vector<int>>> path_info = assemble_paths(mesh, mesh.field(pos_id), intersections);

  // add the path fields to the mesh
  FieldId<int, FaceId> face_intersections_id = mesh.add_field(path_info.x); // index into intersection_indices
  FieldId<int, HalfedgeId> edge_intersections_id = mesh.add_field(path_info.y); // index into intersection_indices
  FieldId<int, VertexId> vertex_intersections_id = mesh.add_field(path_info.z); // index into intersections

  create_intersection_vertices(mesh, pos_id, intersections, vertex_intersections_id);

  Tuple<FieldId<HalfedgeId, HalfedgeId>,
        FieldId<FaceId, FaceId>> seam_info = retriangulate_faces(mesh, pos_id, intersections,
                                                                 path_info.w,
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

#include <geode/python/wrap.h>

using namespace geode;

void wrap_intersections() {
  GEODE_FUNCTION(csg);
}
