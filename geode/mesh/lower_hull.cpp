#include <geode/mesh/lower_hull.h>
#include <geode/mesh/TriangleTopology.h>
#include <geode/structure/UnionFind.h>
#include <geode/exact/mesh_csg.h>
#include <geode/vector/Rotation.h>
#include <geode/array/amap.h>
#include <geode/geometry/platonic.h>

namespace geode {

typedef real T;
typedef Vector<real,3> TV;

struct Toward {
  TriangleTopology const &mesh;
  Field<TV,VertexId> const &pos;

  TV up;
  real min_dot;

  Toward(TriangleTopology const &mesh, Field<TV,VertexId> const &pos, TV up, real min_dot):
    mesh(mesh), pos(pos), up(up), min_dot(min_dot) {}

  inline bool operator()(FaceId id) const {
    return min_dot <= dot(mesh.normal(id, pos), up);
  }

  inline Field<bool, FaceId> as_field() const {
    Field<bool, FaceId> field = mesh.create_compatible_face_field<bool>();
    for (auto f : mesh.faces()) {
      field[f] = this->operator()(f);
    }
    return field;
  }
};

Tuple<Ref<const TriangleSoup>, Array<TV>> lower_hull(TriangleSoup const &imesh, Array<TV> const &pos, const TV up, const T overhang, const T ground_offset) {

  Ref<MutableTriangleTopology> mesh = new_<MutableTriangleTopology>(imesh);
  FieldId<TV,VertexId> pos_id = mesh->add_field(Field<TV,VertexId>(pos), vertex_position_id);

  // color faces
  Field<bool, FaceId> toward = Toward(mesh, mesh->field(pos_id), up, overhang?sin(M_PI/180.*overhang):0.).as_field();

  // make sure we include space for deleted faces
  UnionFind union_find(mesh->faces_.size());

  // group edge-connected components of equal orientation
  for (auto he : mesh->interior_halfedges()) {
    // skip if the opposite is smaller (either we've already seen it or it's a boundary edge)
    if (mesh->reverse(he) < he)
      continue;

    auto f = mesh->faces(he);
    if (toward[f.x] == toward[f.y]) {
      union_find.merge(f.x.idx(), f.y.idx());
    }
  }

  // make component face lists
  Hashtable<int, Array<FaceId>> component_faces;
  for (FaceId face : mesh->faces()) {
    if (toward[face]) {
      int component = union_find.find(face.idx());
      component_faces[component].append(face);
    }
  }

  // for all components that are toward, add faces to our solution mesh
  auto new_mesh = new_<MutableTriangleTopology>();

  int component_idx = 0;
  for (auto p : component_faces) {
    component_idx++;
    auto faces = p.y;
    auto extracted = mesh->extract(faces);
    auto &component_mesh = *extracted.x;

    // split non-manifold vertices
    component_mesh.split_nonmanifold_vertices();

    // compute the boundary loops (before we add the mirrored faces)
    auto boundary_loops = component_mesh.boundary_loops();

    // add vertices and remember the correspondence of vertices (simple offset)
    int voffset = component_mesh.n_vertices();
    for (int i = 0; i < voffset; ++i) {
      VertexId nv = component_mesh.add_vertex();
      assert(nv.idx() == i + voffset);
      // flatten to ground_offset+epsilon
      TV x = component_mesh.field(pos_id)[VertexId(i)];
      TV ground_pos = x + (- dot(x, up) + ground_offset - component_idx) * up;
      component_mesh.field(pos_id)[nv] = ground_pos;
    }

    // add (inverted) faces
    auto oldfaces = component_mesh.faces();
    int foffset = component_mesh.n_faces();
    for (auto f : oldfaces) {
      Vector<VertexId,3> verts = component_mesh.vertices(f);
      FaceId nf = component_mesh.add_face(verts.xzy());
      assert(nf.idx() == f.idx() + foffset);
    }

    // make side faces
    for (auto loop : boundary_loops) {
      for (int i = 0, j = loop.size()-1; i < loop.size(); j = i++) {
        VertexId vi = component_mesh.src(loop[i]);
        VertexId vj = component_mesh.src(loop[j]);
        VertexId vim = VertexId(vi.idx()+voffset);
        VertexId vjm = VertexId(vj.idx()+voffset);
        component_mesh.add_face(vec(vi, vj, vjm));
        component_mesh.add_face(vec(vi, vjm, vim));
      }
    }

    // TODO: move side faces outward (ever so slightly) to avoid slivers

    // TODO: tilt side faces to achieve angle

    // add to the result mesh
    new_mesh->add(component_mesh);
  }

  auto soupmesh = new_mesh->face_triangle_soup();
  auto lower_hull = split_soup(soupmesh.x, new_mesh->field(pos_id).flat, 0);

  // chop the base off at ground_offset, add a box to subtract everything below
  auto faces = lower_hull.x->elements;
  auto vertices = lower_hull.y;

  // make an axis-aligned bbox whose top is at ground_offset, and whose
  // bottom is below ground_offset - component_faces.size(), and whose x/y
  // dimensions include all all points of the mesh, rotated such that up == z.
  // Then rotate that box back such that z == up.
  auto R = Rotation<TV>::from_rotated_vector(up, vec(0, 0, 1.));

  // reduce to 8 points to be considered
  auto corners = bounding_box(vertices).corners();

  // rotate points
  corners.copy(amap([=](TV v){ return R*v; }, corners));

  // compute bbox again, slightly inflated
  auto aabb = bounding_box(corners).thickened(1.);

  // set z min/max
  aabb.max.z = ground_offset;
  aabb.min.z = ground_offset - component_faces.size() - 1;

  // make inverted bbox mesh
  auto box = cube_mesh(aabb.max, aabb.min);

  // rotate back
  R = R.inverse();
  box.y.copy(amap([=](TV v){ return R*v; }, box.y));

  // add
  faces.extend(box.x->elements + vertices.size());
  vertices.extend(box.y);

  // compute the union
  return split_soup(new_<TriangleSoup>(faces), vertices, 0);
}

}

#include <geode/python/wrap.h>

using namespace geode;

void wrap_lower_hull() {
  GEODE_FUNCTION(lower_hull)
}
