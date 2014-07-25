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

void add_vertex_fan(real division_angle, MutableTriangleTopology &mesh,
                    FieldId<Vector<real,3>, VertexId> pos_id,
                    VertexId vi, VertexId vim, VertexId vj, VertexId vjm,
                    real move_by, Vector<real,3> last_normal, Vector<real,3> normal) {

  // last___vi____vj

  auto &pos = mesh.field(pos_id);
  auto center = pos[vim];
  // this corner is convex
  real angle = acos(dot(last_normal, normal));
  int nnew = ceil(angle/division_angle);
  VertexId base = mesh.add_vertices(nnew);

  // move original point
  pos[vim] = center + move_by * last_normal;

  // use proper interpolation for the normals
  auto rot = Rotation<Vector<real,3>>::from_rotated_vector(last_normal, normal);

  // set positions on new points
  for (int i = 1; i <= nnew; ++i) {
    real f = real(i)/nnew;
    Vector<real,3> N = Rotation<Vector<real,3>>::spherical_linear_interpolation(Rotation<Vector<real,3>>(), rot, f) * last_normal;
    pos[VertexId(base.id+i-1)] = center + move_by * N;
  }

  // add the actual mesh by doing a series of edge_split operations

  // for this purpose, we want there to be a face vi-vim-vjm. If that face
  // doesn't exist, the edge vi-vjm should exist, which we flip
  HalfedgeId h = mesh.halfedge(vim, vj);
  if (h.valid()) {
    GEODE_ASSERT(mesh.opposite(h) == vjm);
    GEODE_ASSERT(mesh.is_flip_safe(h));
    h = mesh.reverse(mesh.flip_edge(h));
    GEODE_ASSERT(mesh.src(h) == vi && mesh.dst(h) == vjm);
  } else {
    h = mesh.halfedge(vi, vjm);
    GEODE_ASSERT(h.valid());
  }
  GEODE_ASSERT(mesh.opposite(h) == vim);

  // this is the edge we'll split, it connects vjm and vim
  h = mesh.reverse(mesh.next(h));
  GEODE_ASSERT(mesh.src(h) == vim && mesh.dst(h) == vjm);

  // split with all our new vertices. This splits the edge h, and the new h is
  // in between vjm and the just inserted point.
  for (int i = nnew-1; i >= 0; --i) {
    VertexId v(base.id+i);
    mesh.split_edge(h,v);
    GEODE_ASSERT(mesh.src(h) == vim);
    GEODE_ASSERT(mesh.dst(h) == v);
  }
}

Tuple<Ref<const TriangleSoup>, Array<TV>> lower_hull(TriangleSoup const &imesh, Array<TV> const &X, TV up, const T ground_offset, const T draft_angle, const T division_angle) {

  auto random = new_<Random>(5349);

  up.normalize();

  Ref<MutableTriangleTopology> mesh = new_<MutableTriangleTopology>(imesh);
  FieldId<TV,VertexId> pos_id = mesh->add_field(Field<TV,VertexId>(X), vertex_position_id);

  T cos_division_angle = cos(division_angle);

  // classify faces
  Field<bool, FaceId> toward = Toward(mesh, mesh->field(pos_id), up, sin(draft_angle)).as_field();

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

    // tilt side faces by ofsetting
    for (auto loop : boundary_loops) {

      // compute all normals and store them. normals[i] is the normal of the
      // edge x[i-1]-x[i]
      // find an edge for which we can compute a normal. start is set to the
      // vertex after that edge (i).
      Array<Vector<real,3>> normals(loop.size(), uninit);
      Array<bool> normal_valid(loop.size(), uninit);
      normal_valid.fill(true);
      int start = -1;
      for (int i = 0, j = loop.size()-1; i < loop.size(); j=i++) {
        // j___n[i]___i
        auto vj = loop[j];
        auto vi = loop[i];
        auto eji = component_mesh.field(pos_id)[vi] - component_mesh.field(pos_id)[vj];
        auto ep = eji.projected_orthogonal_to_unit_direction(up).normalized();
        if (ep.sqr_magnitude() == 0) {
          normals[i] = vec(0.,0.,0.);
          normal_valid[i] = false;
        } else {
          normals[i] = cross(up,ep);
          if (start == -1) {
            start = i;
          }
        }
      }

      // if there is none, this loop is a point, and we can simply add a cone
      // with no ill effects. We will therefore just set two adjacent normals to
      // opposite values and let the regular algorithm handle the details.
      if (start == -1) {
        normals[0] = up.unit_orthogonal_vector();
        normals[1] = -normals[0];
        normal_valid[0] = normal_valid[1] = true;
        start = 0;
      }

      // go through the loop, starting with start. last_normal always contains
      // the normal of the last valid edge before our current vertex (i). We try to
      // compute the edge normal for the edge after, and decide whether to split.
      // If the next edge normal cannot be computed, we do not split, but simply
      // continue offsetting in last_normal direction. Once the next edge normal
      // can be computed, we (possibly) add the required split, and reset last_normal.
      int i = start, j = start==loop.size()-1 ? 0 : start+1;
      bool first = true;
      TV last_normal = normals[start];
      while (first || i != start) {

        // last___x_i___n___x_j
        auto vj = loop[j];
        auto vjm = VertexId(vj.idx() + voffset);
        auto vi = loop[i];
        auto vim = VertexId(vi.idx() + voffset);

        auto xi = component_mesh.field(pos_id)[vi];
        auto xim = component_mesh.field(pos_id)[vim];
        auto xj = component_mesh.field(pos_id)[vj];

        // see how far we have to move this point to achieve the draft angle
        auto move_by = tan(draft_angle) * dot(xi-xim, up);

        TV normal = normals[j];

        if (normal_valid[j]) {
          bool convex = Plane<real>(last_normal, xi).phi(xj) < 0;
          bool very_convex = convex && dot(last_normal, normal) < cos_division_angle;

          if (very_convex) {
            // if normal can be computed, and it's too spiky with last_normal, add a fan
            add_vertex_fan(division_angle, component_mesh, pos_id, vi, vim, vj, vjm, move_by, last_normal, normal);
          } else {
            // not spiky, just move point
            component_mesh.field(pos_id)[vim] += move_by * (last_normal+normal).normalized();
          }
          last_normal = normal;
        } else {
          // can't be computed, just move point along last normal
          component_mesh.field(pos_id)[vim] += move_by * last_normal.normalized();
        }

        first = false;
        i = j;
        j++;
        if (j == loop.size())
          j = 0;
      }
    }

    // TODO: move side faces outward (ever so slightly) to avoid slivers
    // They are more likely to be aligned than not.

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
