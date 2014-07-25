#include <geode/mesh/lower_hull.h>
#include <geode/mesh/TriangleTopology.h>
#include <geode/structure/UnionFind.h>
#include <geode/exact/mesh_csg.h>
#include <geode/vector/Rotation.h>
#include <geode/array/amap.h>
#include <geode/geometry/platonic.h>
#include <geode/random/Random.h>

namespace geode {

typedef real T;
typedef Vector<real,3> TV;

struct Toward {
  TriangleTopology const &mesh;
  RawField<const TV,VertexId> X;

  TV up;
  real min_dot;

  Toward(TriangleTopology const &mesh, RawField<const TV,VertexId> X, TV up, real min_dot):
    mesh(mesh), X(X), up(up), min_dot(min_dot) {}

  inline bool operator()(FaceId id) const {
    return min_dot <= dot(mesh.normal(X,id), up);
  }

  inline Field<bool, FaceId> as_field() const {
    Field<bool, FaceId> field = mesh.create_compatible_face_field<bool>();
    for (auto f : mesh.faces()) {
      field[f] = this->operator()(f);
    }
    return field;
  }
};

Tuple<Ref<const TriangleSoup>, Array<TV>> lower_hull(TriangleSoup const &imesh, Array<TV> const &X, TV up, const T ground_offset) {

  auto random = new_<Random>(5349);

  // TODO: this doesn't work properly.
  //T draft_angle = 0.;
  T overhang = 0.;
  up.normalize();

  Ref<MutableTriangleTopology> mesh = new_<MutableTriangleTopology>(imesh);
  FieldId<TV,VertexId> pos_id = mesh->add_field(Field<TV,VertexId>(X), vertex_position_id);

  // classify faces
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
  FieldId<TV,VertexId> new_pos_id = new_mesh->add_vertex_field<TV>(vertex_position_id);
  GEODE_ASSERT(new_pos_id == pos_id);

  // we'll extend downward at least until the minimum value along up, or ground_offset, whichever is lower
  real min_ground = ground_offset;
  for (auto x : mesh->field(pos_id).flat) {
    min_ground = min(min_ground, dot(x, up));
  }

  int component_idx = 0;
  real min_z = min_ground;
  for (auto p : component_faces) {
    component_idx++;
    auto faces = p.y;
    auto extracted = mesh->extract(faces);
    auto &component_mesh = *extracted.x;

    // split non-manifold vertices
    component_mesh.split_nonmanifold_vertices();

    // compute the boundary loops (before we add the mirrored faces)
    // store vertices because the boundary halfedges will change
    auto boundary_loops = amap([&](HalfedgeId h){return component_mesh.src(h);}, component_mesh.boundary_loops());

    // add vertices and remember the correspondence of vertices (simple offset)
    int voffset = component_mesh.n_vertices();
    for (int i = 0; i < voffset; ++i) {
      VertexId nv = component_mesh.add_vertex();
      assert(nv.idx() == i + voffset);
      // flatten to ground_offset+epsilon*component_idx with some randomness to minimize hard cases for subsequent CSG
      TV x = component_mesh.field(pos_id)[VertexId(i)];
      TV ground_pos = x + (- dot(x, up) + min_ground - component_idx - .5 + random->uniform<real>(0., .5)) * up;
      min_z = min(min_z, dot(ground_pos-up, up));
      component_mesh.field(pos_id)[nv] = ground_pos;
    }

    // add (inverted) faces
    auto oldfaces = component_mesh.faces();
    GEODE_DEBUG_ONLY(const int foffset = component_mesh.n_faces();)
    for (auto f : oldfaces) {
      Vector<VertexId,3> verts = Vector<VertexId,3>::map([=](VertexId x){ return VertexId(x.id + voffset); }, component_mesh.vertices(f));
      GEODE_DEBUG_ONLY(const auto nf =) component_mesh.add_face(verts.xzy());
      assert(nf.idx() == f.idx() + foffset);
    }

    // make side faces
    for (auto loop : boundary_loops) {
      for (int i = 0, j = loop.size()-1; i < loop.size(); j = i++) {
        VertexId vi = loop[i];
        VertexId vj = loop[j];
        VertexId vim = VertexId(vi.idx()+voffset);
        VertexId vjm = VertexId(vj.idx()+voffset);
        component_mesh.add_face(vec(vj, vi, vjm));
        component_mesh.add_face(vec(vjm, vi, vim));
      }
    }

    // TODO: move side faces outward (ever so slightly) to avoid slivers

    // TODO: tilt side faces to achieve angle

    // add to the result mesh
    new_mesh->add(component_mesh);
  }

  // all these faces have weight 1
  int weight_one_faces = new_mesh->n_faces();

  // make an axis-aligned bbox whose top is at ground_offset, and whose
  // bottom is below ground_offset - component_faces.size(), and whose x/y
  // dimensions include all all points of the mesh, rotated such that up == z.
  // Then rotate that box back such that z == up.
  auto R = Rotation<TV>::from_rotated_vector(up, vec(0, 0, 1.));

  // reduce to 8 points to be considered, and rotate those
  auto corners = amap([&](TV v){ return R*v; }, bounding_box(new_mesh->field(new_pos_id).flat).corners());

  // compute bbox again, slightly inflated
  auto aabb = bounding_box(corners).thickened(1.);

  // set z min/max
  aabb.max.z = ground_offset;
  aabb.min.z = min_z; //ground_offset - component_faces.size() - 1;

  // make inverted bbox mesh
  auto box = cube_mesh(aabb.max, aabb.min);

  // rotate back
  R = R.inverse();

  // add vertices and set coordinates
  VertexId base = new_mesh->add_vertices(box.y.size());
  for (int i = 0; i < box.y.size(); ++i)
    new_mesh->field(new_pos_id)[VertexId(base.id+i)] = R*box.y[i];

  // add faces
  new_mesh->add_faces((box.x->elements + base.id).copy());

  // make a weight vector
  Array<int> weights(new_mesh->n_faces(), uninit);
  weights.fill(1);
  weights.slice(weight_one_faces, weight_one_faces+box.x->elements.size()).fill(1<<24);

  // compute the union
  auto result = split_soup(new_mesh->face_soup().x, new_mesh->field(new_pos_id).flat, weights, 0);
  return result;
}

}

#include <geode/python/wrap.h>

using namespace geode;

void wrap_lower_hull() {
  GEODE_FUNCTION(lower_hull)
}
