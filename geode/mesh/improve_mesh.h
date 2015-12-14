#pragma once

#include <geode/mesh/TriangleTopology.h>
#include <geode/mesh/quadric.h>
#include <geode/math/lerp.h>
#include <geode/solver/powell.h>

namespace geode {

struct ImproveOptions {

  ImproveOptions(real min_quality, real max_distance, real max_silhouette_distance, real min_normal_dot = .8, real max_iter = 30, bool can_change_topology = true, real min_relevant_area = 1e-12, real min_quality_improvement = 1e-6)
  : min_quality(min_quality)
  , max_distance(max_distance)
  , max_silhouette_distance(max_silhouette_distance)
  , min_normal_dot(min_normal_dot)
  , max_iter(max_iter)
  , can_change_topology(can_change_topology)
  , min_relevant_area(min_relevant_area)
  , min_quality_improvement(min_quality_improvement)
  {}

  real min_quality;
  real max_distance;
  real max_silhouette_distance;
  real min_normal_dot;
  int max_iter;
  bool can_change_topology;
  real min_relevant_area;
  real min_quality_improvement;
};

Quadric compute_silhouette_quadric(TriangleTopology const &mesh, Field<Vector<real,3>, VertexId> const &pos, VertexId v, real min_relevant_area) {
  // for boundary vertices, add planes perpendcular to incident boundary faces, containing
  // boundary edges
  // for non-boundary faces, add planes perpendicular to incident faces with high
  // dihedral angles, containing edges with high dihedral angles
  // for non-boundary vertices, the perpendicular planes only become relevant once the
  // angle is greater than 90 degrees. At 90 degrees, both silhouette and regular quadrics
  // do pretty much the same thing.
  Quadric q;
  real total = 0;

  FaceId last_face;
  HalfedgeId last_boundary_edge;
  bool have_last_normal = false;
  HalfedgeId start_e = mesh.halfedge(v);
  HalfedgeId end_e, e = start_e;

  //GEODE_DEBUG_ONLY(std::cout << "computing silhouette quadric for " << v << ", valence " << mesh.valence(v) << ", relevant area " << min_relevant_area << std::endl);

  while (e != end_e) {
    // the first time we set normal, remember where we were.
    if (!end_e.valid() && have_last_normal) {
      end_e = e;
    }

    //GEODE_DEBUG_ONLY(std::cout << "  considering edge " << e << ", boundary " << mesh.is_boundary(e) << ", have_last " << have_last_normal << std::endl);
    Vector<real,3> edge_vector = mesh.segment(pos, e).vector();
    if (mesh.is_boundary(e)) {

      // if we have last normal, add a plane for this boundary edge
      if (have_last_normal) {
        Vector<real,3> last_normal = mesh.normal(pos, last_face);
        real w = q.add_plane(cross(last_normal, edge_vector), pos[v]);
        total += w;
        //GEODE_DEBUG_ONLY(std::cout << "    boundary: normal " << last_normal << ", edge " << edge_vector << ", weight " << w << std::endl);
      }

      // remember the other boundary edge
      last_boundary_edge = mesh.prev(e);

      have_last_normal = false;
    } else {

      FaceId face = mesh.face(e);
      if (mesh.area(pos, face) > min_relevant_area) {
        Vector<real,3> normal = mesh.normal(pos,face);

        // check if we still need to process a boundary
        if (last_boundary_edge.valid()) {
          Vector<real,3> last_edge_vector = mesh.segment(pos, last_boundary_edge).vector();
          real w = q.add_plane(cross(normal, last_edge_vector), pos[v]);
          total += w;
          last_boundary_edge = HalfedgeId();
          //GEODE_DEBUG_ONLY(std::cout << "    back boundary: normal " << normal << ", edge " << last_edge_vector << ", weight " << w << std::endl);
        }

        // check dihedral with last face, and add plane
        if (have_last_normal) {
          Vector<real,3> last_normal = mesh.normal(pos, last_face);
          if (dot(normal, last_normal) < 0) {
            auto n = .5 * (cross(normal, edge_vector) - cross(last_normal, edge_vector));
            real w = q.add_plane(n, pos[v]);
            total += w;
            //GEODE_DEBUG_ONLY(std::cout << "    dihedral: normal dot " << dot(normal, last_normal) << " le " << edge_vector.magnitude() << " normals " << last_normal << " " << normal << ", crosses " << cross(last_normal, edge_vector) << " " << cross(normal, edge_vector) << ", edge " << edge_vector << ", n " << n << ", weight " << w << std::endl);
          }
        }
        last_face = face;
        have_last_normal = true;
      }
    }

    e = mesh.left(e);

    // it may be that we never set end_e
    if (e == start_e && !end_e.valid()) {
      //GEODE_DEBUG_ONLY(std::cout << "  haven't found a valid face, no silhouette." << std::endl);
      break;
    }
  }
  q.normalize(total);
  return q;
}

template<class Quality>
real mesh_quality(TriangleTopology const &mesh, Field<Vector<real,3>, VertexId> const &pos, Quality &Q) {
  real minq = 1;
  for (auto f : mesh.faces()) {
    minq = min(minq, Q(mesh.triangle(pos, f)));
  }
  return minq;
}

inline real mesh_quality(TriangleTopology const &mesh, Field<Vector<real,3>, VertexId> const &pos) {
  real minq = 1;
  for (auto f : mesh.faces()) {
    minq = min(minq, mesh.triangle(pos, f).quality());
  }
  return minq;
}

template<class Quality>
real mesh_quality(MutableTriangleTopology const &mesh, Quality &Q) {
  FieldId<Vector<real,3>,VertexId> pos_id(vertex_position_id);
  auto &pos = mesh.field(pos_id);
  real minq = 1;
  for (auto f : mesh.faces()) {
    minq = min(minq, Q(mesh.triangle(pos, f)));
  }
  return minq;
}

inline real mesh_quality(MutableTriangleTopology const &mesh) {
  FieldId<Vector<real,3>,VertexId> pos_id(vertex_position_id);
  auto &pos = mesh.field(pos_id);
  real minq = 1;
  for (auto f : mesh.faces()) {
    minq = min(minq, mesh.triangle(pos, f).quality());
  }
  return minq;
}

// Quality takes a Triangle and returns a normalized quality in [0,1]. The output mesh
// has no triangles with Q(triangle(i)) < min_quality if the function succeeds
// VertexLocked has an operator(), If VL(v) returns false,
// a vertex cannot be moved and no outgoing halfedge can be contracted.
// EdgeLocked takes two vertices, if EL(v0,v1) returns false, the edge between v0 and v1
// cannot be flipped or contracted (it's done on vertices to avoid problems with
// edges changing id in flips).
// max_distance is the maximum distance (as estimated by quadrics) that a vertex can move from
// the original surface
// min_normal_dot is the minimum dot product of any face normal before and after a transformation
// if a transformation changes any face normal more than this, it is not allowed.
// For boundary edges, we apply the same criterion to implicitly defined planes perpendicular to
// the incident face.

template<class Quality, class EdgeLocked, class VertexLocked>
struct ImproveHelper {
  MutableTriangleTopology &mesh;
  Field<Vector<real,3>, VertexId> const &pos; // this is definitely registered with the mesh
  Field<real, FaceId> &quality; // these are not automatically resized, so we take care of them.
  Field<Quadric, VertexId> &quadrics;
  Field<Quadric, VertexId> &silhouette_quadrics;
  Quality &Q;
  EdgeLocked &EL;
  VertexLocked &VL;
  int original_vertex_count;
  ImproveOptions const &o;

  ImproveHelper(MutableTriangleTopology &mesh, Field<Vector<real,3>, VertexId> const &pos,
                Field<real, FaceId> &quality, Field<Quadric, VertexId> &quadrics,
                Field<Quadric, VertexId> &silhouette_quadrics,
                Quality &Q, EdgeLocked &EL, VertexLocked &VL,
                ImproveOptions const &o)
  : mesh(mesh)
  , pos(pos)
  , quality(quality)
  , quadrics(quadrics)
  , silhouette_quadrics(silhouette_quadrics)
  , Q(Q)
  , EL(EL)
  , VL(VL)
  , o(o)
  {}

  // costs are in [0,1), or infinity
  real quadric_cost(real quadric_error) {
    if (quadric_error >= o.max_distance*o.max_distance)
      return numeric_limits<real>::infinity();
    return quadric_error/(o.max_distance*o.max_distance);
  }
  real area_cost(real old_area, real new_area) {
    // if we're increasing or decreasing the area by a lot, there's likely a problem.
    if (abs(new_area - old_area) < o.min_relevant_area)
      return 0.;
    real diff = abs(new_area - old_area);
    real permitted_area_change = max(0., old_area / o.min_normal_dot - old_area);
    if (diff > permitted_area_change)
      return numeric_limits<real>::infinity();
    return diff/permitted_area_change;
  }
  real normal_cost(real normal_dot) {
    if (normal_dot <= o.min_normal_dot)
      return numeric_limits<real>::infinity();
    return lerp(normal_dot, o.min_normal_dot, 1., 1., 0.);
  }
  real silhouette_cost(real quadric_error) {
    if (quadric_error >= o.max_silhouette_distance*o.max_silhouette_distance)
      return numeric_limits<real>::infinity();
    return quadric_error/(o.max_silhouette_distance*o.max_silhouette_distance);
  }

  // criteria:
  // - cannot move vertices further than allowed by quadrics
  // - cannot change area by more than is sensible given normal constraints
  // - when changing a set of triangles, cannot decrease the quality of any triangle
  //   below the worst quality going in
  // - finite costs for operations are the maximum normal and quadric cost of any changed
  //   vertex/face
  // - cannot violate locked edges/vertices
  // These functions return a tuple with
  // - a cost (if this cost is infinity, the other return fields may be empty)
  // - an array of (face ID, new_quality) tuples of all faces that will be changed
  //   by this operation, and their qualities after the operation (all vertices
  //   incident to any of these faces should have their quadrics recomputed)
  // The cost is +inf if the operation is not allowed, and a finite cost otherwise.

  Tuple<real,Array<Tuple<FaceId,real>>> check_flip(HalfedgeId h) {

    if (EL(mesh.src(h), mesh.dst(h)))
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>());

    if (!mesh.is_flip_safe(h))
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>());

    // old faces
    HalfedgeId r = mesh.reverse(h);
    FaceId f1 = mesh.face(h);
    FaceId f2 = mesh.face(r);
    auto old_t1 = mesh.triangle(pos, f1);
    auto old_t2 = mesh.triangle(pos, f2);
    real old_q1 = quality[f1];
    real old_q2 = quality[f2];

    // new faces
    auto new_t1 = Triangle<Vector<real,3>>(pos[mesh.dst(h)], pos[mesh.opposite(h)], pos[mesh.opposite(r)]);
    auto new_t2 = Triangle<Vector<real,3>>(pos[mesh.dst(r)], pos[mesh.opposite(r)], pos[mesh.opposite(h)]);
    real new_q1 = Q(new_t1);
    real new_q2 = Q(new_t2);

    if (o.min_quality_improvement+min(old_q1, old_q2) >= min(new_q1, new_q2)) {
      //GEODE_DEBUG_ONLY(std::cout << "    illegal flip for " << mesh.src(h) << " - " << mesh.dst(h) << ", q " << min(old_q1, old_q2) << " -> " << min(new_q1, new_q2) << ", improvement " << min(new_q1, new_q2)-min(old_q1, old_q2) << std::endl);
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>());
    }

    real old_a1 = old_t1.area();
    real old_a2 = old_t2.area();
    auto old_n1 = old_t1.normal();
    auto old_n2 = old_t2.normal();
    real new_a1 = new_t1.area();
    real new_a2 = new_t2.area();
    auto new_n1 = new_t1.normal();
    auto new_n2 = new_t2.normal();

    // compute minimum normal dot product between any pair of old and new faces
    // for which a normal can be computed
    real min_dot = 1;

    if (old_a1 > o.min_relevant_area && new_a1 > o.min_relevant_area)
      min_dot = min(min_dot, dot(old_n1, new_n1));
    if (old_a2 > o.min_relevant_area && new_a1 > o.min_relevant_area)
      min_dot = min(min_dot, dot(old_n2, new_n1));
    if (old_a1 > o.min_relevant_area && new_a2 > o.min_relevant_area)
      min_dot = min(min_dot, dot(old_n1, new_n2));
    if (old_a2 > o.min_relevant_area && new_a2 > o.min_relevant_area)
      min_dot = min(min_dot, dot(old_n2, new_n2));

    Array<Tuple<FaceId,real>> changed;
    changed.append(tuple(f1, new_q1));
    changed.append(tuple(f2, new_q2));

    GEODE_DEBUG_ONLY(std::cout << "    checking flip for " << mesh.src(h) << " - " << mesh.dst(h) << ", dot: " << min_dot << " (areas " << old_a1 << ", " << old_a2 << " -> " << new_a1 << ", " << new_a2 << "), q " << min(old_q1, old_q2) << " -> " << min(new_q1, new_q2) << ", improvement " << min(new_q1, new_q2)-min(old_q1, old_q2) << std::endl);

    return tuple(normal_cost(min_dot)+area_cost(old_a1+old_a2, new_a1+new_a2), changed);
  }

  // these update the locked fields where necessary
  void do_flip(HalfedgeId he) {
    he = mesh.flip_edge(he);
  }

  void do_collapse(HalfedgeId he) {
    mesh.collapse(he);
  }

  Array<Tuple<FaceId,real>> do_split(HalfedgeId he, Vector<real,3> const &to) {
    // split
    VertexId nv = mesh.split_edge(he);
    // move
    pos[nv] = to;

    // adjust vertex fields
    quadrics.append(compute_quadric(mesh, pos, nv));
    silhouette_quadrics.append(compute_silhouette_quadric(mesh, pos, nv, o.min_relevant_area));

    // return new faces and qualities
    Array<Tuple<FaceId,real>> new_faces;
    for (auto f : mesh.incident_faces(nv)) {
      quality.grow_until_valid(f);
      quality[f] = Q(mesh.triangle(pos,f));
      new_faces.append(tuple(f, quality[f]));
    }

    return new_faces;
  }

  void do_move(VertexId v, Vector<real,3> const &to) {
    pos[v] = to;
  }

  real check_twin_face(FaceId f) {

    // check if this is a twin face
    if (mesh.is_boundary(f) ||
        mesh.opposite(mesh.reverse(mesh.halfedge(f,0))) != mesh.opposite(mesh.halfedge(f,0))) {
      return numeric_limits<real>::infinity();
    }

    // check if it's ok to remove
    const auto vs = mesh.vertices(f);

    bool locked = false;
    for (auto h : mesh.halfedges(f)) {
      locked = locked || EL(mesh.src(h), mesh.dst(h)) || EL(mesh.dst(h), mesh.src(h));
    }

    for (auto v : vs) {
      locked = locked || VL(v);
    }

    if (locked)
      return numeric_limits<real>::infinity();

    // compute silhouette cost (we're collapsing all edges, all other costs are
    // zero or irrelevant)
    return max(silhouette_cost(silhouette_quadrics[vs.x](pos[vs.y])),
           max(silhouette_cost(silhouette_quadrics[vs.y](pos[vs.z])),
               silhouette_cost(silhouette_quadrics[vs.z](pos[vs.x]))));
  }

  // check if we can fill in a tunnel (we should check all halfedges for this,
  // not only those next to problematic faces)
  Tuple<real,Array<Tuple<FaceId,real>>> check_tunnel(HalfedgeId h) {
    Vector<VertexId,3> vs(mesh.src(h), mesh.dst(h), mesh.tunnel_vertex(h));

    if (!vs.z.valid()) {
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>());
    }

    // We have a tunnel. Can we fill it in?
    // We can't if any of the edges are locked, or any of the vertices are locked,
    // or if the silhouette would change too much. We'll measure the last effect
    // using the quadrics and silhouette quadrics pretending to collapse each of
    // the edges.
    bool locked = false;
    for (int i = 0, j = 2; i < 3; j = i++) {
      locked = locked || EL(vs[i], vs[j]) || EL(vs[j], vs[i]);
    }
    for (auto v : vs) {
      locked = locked || VL(v);
    }

    if (locked)
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>());

    return max(quadric_cost(quadrics[vs.x](pos[vs.y])),
           max(quadric_cost(quadrics[vs.y](pos[vs.z])),
               quadric_cost(quadrics[vs.z](pos[vs.x]))))
         + max(silhouette_cost(silhouette_quadrics[vs.x](pos[vs.y])),
           max(silhouette_cost(silhouette_quadrics[vs.y](pos[vs.z])),
               silhouette_cost(silhouette_quadrics[vs.z](pos[vs.x]))));
  }

  Tuple<real,Array<Tuple<FaceId,real>>> check_collapse(HalfedgeId h) {
    VertexId v0 = mesh.src(h);
    VertexId v1 = mesh.dst(h);

    bool edge_locked = false;
    for (auto h : mesh.outgoing(v0)) {
      edge_locked = edge_locked || EL(mesh.src(h), mesh.dst(h)) || EL(mesh.dst(h), mesh.src(h));
    }

    if (edge_locked || VL(v0))
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>());

    if (!mesh.is_collapse_safe(h)) {
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>());
    }

    // compute quadric cost
    real quadric_cost = this->quadric_cost(quadrics[v0](pos[v1]));
    if (quadric_cost == numeric_limits<real>::infinity()) {
      //GEODE_DEBUG_ONLY(std::cout << "    illegal collapse " << mesh.src(h) << " -> " << mesh.dst(h) << ", bad quadric: " << quadrics[v0](pos[v1]) << std::endl);
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>());
    }

    // compute silhouette cost
    real silhouette_cost = this->silhouette_cost(silhouette_quadrics[v0](pos[v1]));
    if (silhouette_cost == numeric_limits<real>::infinity()) {
      //GEODE_DEBUG_ONLY(std::cout << "    illegal collapse " << mesh.src(h) << " -> " << mesh.dst(h) << ", bad silhouette quadric: " << silhouette_quadrics[v0](pos[v1]) << std::endl);
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>());
    }

    // compute minimum quality before
    real minq_before = 1;
    for (auto f : mesh.incident_faces(v0)) {
      minq_before = min(minq_before, quality[f]);
    }

    Array<Tuple<FaceId,real>> new_faces;
    real max_normal_cost = 0;
    real old_area = 0, new_area = 0;
    for (auto f : mesh.incident_faces(v0)) {
      if (mesh.vertices(f).contains(v1))
        continue;

      // get current normal
      auto t = mesh.triangle(pos, f);
      real old_a = t.area();
      old_area += old_a;
      auto old_n = t.normal();

      // compute new triangles, normals, qualities
      auto vs = mesh.vertices(f);
      for (auto &v : vs)
        if (v == v0) {
          v = v1; break;
        }
      auto new_t = Triangle<Vector<real,3>>(pos[vs[0]], pos[vs[1]], pos[vs[2]]);
      real new_q = Q(new_t);
      real new_a = new_t.area();
      new_area += new_a;
      auto new_n = new_t.normal();

      // compute whether quality allows this operation
      if (new_q < minq_before && minq_before > o.min_quality_improvement) { // if both are terrible, removing the triangle is better than not removing it
        //GEODE_DEBUG_ONLY(std::cout << "    illegal collapse " << mesh.src(h) << " -> " << mesh.dst(h) << ", quadric: " << quadrics[v0](pos[v1]) << ", quality decreases by at least " << - new_q + minq_before << " (from " << minq_before << " > " << o.min_quality_improvement << ")" << std::endl);
        return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>());
      }

      // compute normal cost
      if (old_a > o.min_relevant_area && new_a > o.min_relevant_area) {
        max_normal_cost = max(max_normal_cost, normal_cost(dot(old_n, new_n)));

        assert(max_normal_cost >= 0);

        if (max_normal_cost == numeric_limits<real>::infinity()) {
          //GEODE_DEBUG_ONLY(std::cout << "    illegal collapse " << mesh.src(h) << " -> " << mesh.dst(h) << ", quadric: " << quadrics[v0](pos[v1]) << ", bad normal dot " << dot(old_n, new_n) << ", areas " << old_a << ", " << new_a << std::endl);
          return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>());
        }
      }

      // remember modified faces and new qualities
      new_faces.append(tuple(f, new_q));
    }

    GEODE_DEBUG_ONLY(std::cout << "    checking collapse " << mesh.src(h) << " -> " << mesh.dst(h) << ", quadric: " << quadrics[v0](pos[v1]) << ", silhouette quadric: " << silhouette_quadrics[v0](pos[v1]) << ", normal_cost " << max_normal_cost << ", area cost " << area_cost(old_area, new_area) << " (" << old_area << " -> " << new_area << ")" << std::endl);

    return tuple(silhouette_cost + quadric_cost + max_normal_cost + area_cost(old_area, new_area), new_faces);
  }

  // returns min quality after and the optimal position. The invalid
  // vertex id in each face is replaced with the moved vertex position.
  Tuple<real, Vector<real,3>> optimize_position(Array<Vector<VertexId,3>> faces, Vector<real,3> xinit, Quadric const &q, Quadric const &sq) {
    real scale = 0;
    Array<real> areas;
    Array<Vector<real,3>> normals;
    for (auto f : faces) {
      Triangle<Vector<real,3>> t(f[0].valid() ? pos[f[0]] : xinit,
                                 f[1].valid() ? pos[f[1]] : xinit,
                                 f[2].valid() ? pos[f[2]] : xinit);
      areas.append(t.area());
      normals.append(t.normal());

      // scale: mean edge length
      auto el = t.edge_lengths();
      scale += el[0] + el[1] + el[2];
    }
    scale /= faces.size() * 3;

    if (!scale)
      return tuple(-1., Vector<real,3>());

    auto objective = [&](RawArray<const real> x) {
      auto p = vec(x[0], x[1], x[2]);

      // minimize (1 - min_quality), or >1 if any of the normals move
      // too much, or if the quadric error is too high
      real min_q = 1;
      int i = 0;
      for (auto f : faces) {
        Triangle<Vector<real,3>> t(f[0].valid() ? pos[f[0]] : p,
                                   f[1].valid() ? pos[f[1]] : p,
                                   f[2].valid() ? pos[f[2]] : p);

        // check quadrics
        if (q(p) > o.max_distance*o.max_distance)
          return 1. + q(p) - o.max_distance*o.max_distance; // make the objective smooth

        // check silhouette quadrics
        if (sq(p) > o.max_silhouette_distance*o.max_silhouette_distance)
          return 1. + sq(p) - o.max_silhouette_distance*o.max_silhouette_distance; // make the objective smooth

        // compute quality
        min_q = min(min_q, Q(t));

        // check normal
        auto a = t.area();
        auto n = t.normal();

        auto old_a = areas[i];
        auto old_n = normals[i];

        if (old_a > o.min_relevant_area && a > o.min_relevant_area)
          if (dot(n,old_n) < o.min_normal_dot)
            return 1. + o.min_normal_dot - dot(n,old_n); // make the objective smooth

        i++;
      }
      return 1 - min_q;
    };

    Array<real> pa;
    pa.copy(xinit);
    auto res = powell(objective, pa, scale/5, scale/100, .001, 20);
    auto qafter = 1-res.x;
    return tuple(qafter, vec(pa[0], pa[1], pa[2]));
  }

  // this function additionally returns the optimal (while still allowed) position to move to
  Tuple<real,Array<Tuple<FaceId,real>>, Vector<real,3>> check_move(VertexId v) {
    bool edge_locked = false;
    for (auto h : mesh.outgoing(v)) {
      edge_locked = edge_locked || EL(mesh.src(h), mesh.dst(h)) || EL(mesh.dst(h), mesh.src(h));
    }

    if (edge_locked || VL(v))
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>(), Vector<real,3>());

    // TODO: we should actually put a proper constrained optimization here:
    // optimize the minimum incident triangle quality, subject quadric constraints
    // on the vertex and normal constraints on the incident faces.

    // For now, just use Powell, it's probably good enough.
    auto faces = mesh.incident_faces(v);
    Array<Vector<VertexId,3>> face_verts;
    real qbefore = 1.;
    for (auto f : faces) {
      qbefore = min(qbefore, quality[f]);
      auto verts = mesh.vertices(f);
      verts[verts.find(v)] = VertexId();
      face_verts.append(verts);
    }

    auto res = optimize_position(face_verts, pos[v], quadrics[v], silhouette_quadrics[v]);
    real qafter = res.x;
    auto p = res.y;

    if (qafter < qbefore+o.min_quality_improvement) { // illegal or no (tangible) improvement?
      //GEODE_DEBUG_ONLY(std::cout << "    illegal move: p-x " << p-pos[v] << ", q " << qbefore << " -> " << qafter << ", improvement " << qafter-qbefore << std::endl);
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>(), Vector<real,3>());
    }

    Array<Tuple<FaceId,real>> new_faces;
    real min_dot = 1;
    real old_area = 0, new_area = 0;
    GEODE_DEBUG_ONLY(real minq = 1);
    int i = 0;
    for (auto f : faces) {
      auto verts = mesh.vertices(f);
      auto old_t = mesh.triangle(pos,f);

      int k = verts.find(v);
      Triangle<Vector<real,3>> t(k == 0 ? p : pos[verts[0]],
                                 k == 1 ? p : pos[verts[1]],
                                 k == 2 ? p : pos[verts[2]]);
      new_faces.append(tuple(f, Q(t)));

      // check normal
      auto a = t.area();
      new_area += a;
      auto n = t.normal();

      auto old_a = old_t.area();
      old_area += old_a;
      auto old_n = old_t.normal();

      if (old_a > o.min_relevant_area && a > o.min_relevant_area)
        min_dot = min(min_dot, dot(n,old_n));

      i++;

      GEODE_DEBUG_ONLY(minq = min(minq, new_faces.back().y));
    }

    GEODE_DEBUG_ONLY(GEODE_ASSERT(fabs(minq - qafter) < 1e-12));
    GEODE_DEBUG_ONLY(std::cout << "    checking move: p-x " << p-pos[v] << ", cost " << quadric_cost(quadrics[v](p)) + silhouette_cost(silhouette_quadrics[v](p)) + normal_cost(min_dot) + area_cost(old_area, new_area) << ", q " << qbefore << " -> " << minq << ", improvement " << minq-qbefore << std::endl);

    return tuple(quadric_cost(quadrics[v](p)) + silhouette_cost(silhouette_quadrics[v](p)) + normal_cost(min_dot) + area_cost(old_area, new_area), new_faces, p);
  }

  // check if an edge can be split. This splits the edge and optimizes the new vertex position
  Tuple<real,Array<Tuple<FaceId,real>>, Vector<real,3>> check_split(HalfedgeId h) {
    VertexId v0 = mesh.src(h);
    VertexId v1 = mesh.dst(h);

    // can't split locked edges
    if (EL(v0, v1) || EL(v1, v0))
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>(), Vector<real,3>());

    Quadric q, sq;
    real total_weight = 0;
    real qbefore = 1.;
    Array<Vector<VertexId,3>> face_verts;
    auto faces = mesh.faces(h);
    for (auto f : faces) {
      // compute quality before
      if (!f.valid())
        continue;

      qbefore = min(qbefore, quality[f]);

      // make new faces with new virtual vertex
      auto verts = mesh.vertices(f);
      auto verts2 = verts;

      verts[verts.find(v0)] = VertexId();
      verts2[verts.find(v1)] = VertexId();
      face_verts.append(verts);
      face_verts.append(verts2);

      // make pretend quadrics for new vertex. Since the new vertex splits the
      // incident faces evenly, we can just add the two original faces.
      total_weight += q.add_face(mesh.triangle(pos, f));
    }
    q.normalize(total_weight);

    // make silhouette quadric
    Vector<real,3> edge_vector = mesh.segment(pos, h).vector();
    if (faces.x.valid() && faces.y.valid() &&
        mesh.area(pos, faces.x) > o.min_relevant_area &&
        mesh.area(pos, faces.y) > o.min_relevant_area) {
      Vector<real,3> n1 = mesh.normal(pos, faces.x);
      Vector<real,3> n2 = mesh.normal(pos, faces.y);
      if (dot(n1,n2) < 0) {
        auto n = .5 * (cross(n1, edge_vector) - cross(n2, edge_vector));
        sq.normalize(sq.add_plane(n, pos[v0]));
      }
    } else if (faces.x.valid() && mesh.area(pos, faces.x) > o.min_relevant_area) {
      // this is a boundary, add a single plane perpendicular to the other face
      Vector<real,3> n = cross(mesh.normal(pos, faces.x), edge_vector);
      sq.normalize(sq.add_plane(n,pos[v0]));
    } else if (faces.y.valid() && mesh.area(pos, faces.y) > o.min_relevant_area) {
      // this is a boundary, add a single plane perpendicular to the other face
      Vector<real,3> n = cross(mesh.normal(pos, faces.y), edge_vector);
      sq.normalize(sq.add_plane(n,pos[v0]));
    }

    // optimize position
    auto pinit = .5*(pos[v0] + pos[v1]);
    auto res = optimize_position(face_verts, pinit, q, sq);
    real qafter = res.x;
    auto p = res.y;

    if (qafter < qbefore+o.min_quality_improvement) { // illegal or no (tangible) improvement?
      //GEODE_DEBUG_ONLY(std::cout << "    illegal move: p-x " << p-pos[v] << ", q " << qbefore << " -> " << qafter << ", improvement " << qafter-qbefore << std::endl);
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>(), Vector<real,3>());
    }

    real min_dot = 1;
    real old_area = 0, new_area = 0;
    GEODE_DEBUG_ONLY(real minq = 1);
    int i = 0;
    for (auto f : face_verts) {
      Triangle<Vector<real,3>> old_t(f[0].valid() ? pos[f[0]] : pinit,
                                     f[1].valid() ? pos[f[1]] : pinit,
                                     f[2].valid() ? pos[f[2]] : pinit);

      Triangle<Vector<real,3>> t(f[0].valid() ? pos[f[0]] : p,
                                 f[1].valid() ? pos[f[1]] : p,
                                 f[2].valid() ? pos[f[2]] : p);

      // check normal
      auto a = t.area();
      new_area += a;
      auto n = t.normal();

      auto old_a = old_t.area();
      old_area += old_a;
      auto old_n = old_t.normal();

      if (old_a > o.min_relevant_area && a > o.min_relevant_area)
        min_dot = min(min_dot, dot(n,old_n));

      i++;

      GEODE_DEBUG_ONLY(minq = min(minq, Q(t)));
    }

    real cost = quadric_cost(q(p)) + silhouette_cost(sq(p)) + normal_cost(min_dot) + area_cost(old_area, new_area);

    GEODE_DEBUG_ONLY(GEODE_ASSERT(fabs(minq - qafter) < 1e-12));
    GEODE_DEBUG_ONLY(std::cout << "    checking split: cost " << cost << ", q " << qbefore << " -> " << minq << ", improvement " << minq-qbefore << std::endl);

    return tuple(cost, Array<Tuple<FaceId, real>>(), p);
  }
};

template<class Quality, class EdgeLocked, class VertexLocked>
bool improve_mesh_inplace(MutableTriangleTopology &mesh, FieldId<Vector<real,3>, VertexId> pos_id,
                          ImproveOptions const &o,
                          EdgeLocked &EL, VertexLocked &VL, Quality &Q) {
  GEODE_DEBUG_ONLY(std::cout.precision(18));

  auto &pos = mesh.field(pos_id);

  // cache face qualities
  Field<real,FaceId> quality = mesh.create_compatible_face_field<real>();
  real old_mesh_quality = 1;
  Hashtable<FaceId> needs_improvement;
  for (auto f : mesh.faces()) {
    quality[f] = Q(mesh.triangle(pos, f));
    if (quality[f] < o.min_quality) {
      old_mesh_quality = min(old_mesh_quality, quality[f]);
      needs_improvement.set(f);
    }
  }

  // cache quadrics
  Field<Quadric,VertexId> quadrics = mesh.create_compatible_vertex_field<Quadric>();
  Field<Quadric,VertexId> silhouette_quadrics = mesh.create_compatible_vertex_field<Quadric>();
  for (auto v : mesh.vertices()) {
    quadrics[v] = compute_quadric(mesh, pos, v);
    silhouette_quadrics[v] = compute_silhouette_quadric(mesh, pos, v, o.min_relevant_area);
  }

  ImproveHelper<Quality, EdgeLocked, VertexLocked> helper(mesh, pos, quality, quadrics, silhouette_quadrics, Q, EL, VL, o);

  bool improved_something = !needs_improvement.empty();
  int iter = 0;
  while (improved_something && iter < o.max_iter) {

    GEODE_DEBUG_ONLY(real quality_before = mesh_quality(mesh, pos, Q));

    improved_something = false;
    Hashtable<FaceId> still_needs_improvement;

    for (auto f : needs_improvement) {
      // check if this face has been deleted by another operation
      if (mesh.erased(f))
        continue;

      // check if the quality has been changed by another operation, and if it's
      // still in need of improvement
      real q = quality[f];
      if (q >= o.min_quality)
        continue;

      // find best operation to perform on this triangle we can:
      //   - flip an edge
      //   - collapse an edge (any halfedge outgoing from any triangle vertex)
      //   - move a vertex
      // prioritize the operation which
      //   - improves the quality of the mesh the most (enough)
      //   - has the lowest impact on normals and quadrics
      //   - changes the fewest triangles
      // Priority:
      //   - if there are any operations with finite cost that improve our triangle
      //     above min_quality without pulling any other triangle below, only consider those
      //   - if there are any operations with finite cost that improve our triangle
      //     above min_quality without worsening other triangles, only consider those
      //   - from the leftover operations, pick the one with lowest cost

      Array<Tuple<FaceId,real>> changed_faces;
      real best_cost = numeric_limits<real>::infinity();

      GEODE_DEBUG_ONLY(std::cout << "  face " << f << " quality " << q << std::endl);

      // check flips
      HalfedgeId flip_edge;
      for (auto he : mesh.halfedges(f)) {
        auto r = helper.check_flip(he);
        if (r.x < best_cost) {
          best_cost = r.x;
          flip_edge = he;
          changed_faces = r.y;
        }
      }

      // check collapses
      HalfedgeId collapse_edge;
      for (auto v : mesh.vertices(f)) {
        for (auto he : mesh.outgoing(v)) {
          auto r = helper.check_collapse(he);
          if (r.x < best_cost) {
            flip_edge = HalfedgeId();
            best_cost = r.x;
            collapse_edge = he;
            changed_faces = r.y;
          }
        }
      }

      // if nothing else worked, check vertex moves and splits (they're expensive)
      VertexId move_vertex;
      HalfedgeId split_edge;
      Vector<real,3> move_to;
      if (!flip_edge.valid() && !collapse_edge.valid()) {
        for (auto v : mesh.vertices(f)) {
          auto r = helper.check_move(v);
          if (r.x < best_cost) {
            collapse_edge = flip_edge = HalfedgeId();
            best_cost = r.x;
            move_vertex = v;
            move_to = r.z;
            changed_faces = r.y;
          }
        }

        if (!move_vertex.valid()) {
          for (auto e : mesh.halfedges(f)) {
            auto r = helper.check_split(e);
            if (r.x < best_cost) {
              collapse_edge = flip_edge = HalfedgeId();
              move_vertex = VertexId();
              best_cost = r.x;
              split_edge = e;
              move_to = r.z;
            }
          }
        }
      }

      GEODE_DEBUG_ONLY(real minq_before = 1);

      // do it and update qualities and quadrics
      if (flip_edge.valid()) {
        GEODE_DEBUG_ONLY(
          for (auto bf: mesh.faces(flip_edge)) {
            minq_before = min(minq_before, quality[bf]);
          }
          std::cout << "    flip " << flip_edge << ", cost " << best_cost << std::endl;
        )
        helper.do_flip(flip_edge);
      } else if (collapse_edge.valid()) {
        GEODE_DEBUG_ONLY(
          for (auto bf: mesh.incident_faces(mesh.src(collapse_edge))) {
            minq_before = min(minq_before, quality[bf]);
          }
          std::cout << "    collapse " << collapse_edge << ": " << mesh.src(collapse_edge) << " -> " << mesh.dst(collapse_edge) << ", cost " << best_cost << std::endl;
        )
        helper.do_collapse(collapse_edge);
      } else if (move_vertex.valid()) {
        GEODE_DEBUG_ONLY(
          for (auto bf: mesh.incident_faces(move_vertex)) {
            minq_before = min(minq_before, quality[bf]);
          }
          std::cout << "    move " << move_vertex << ", cost " << best_cost << std::endl;
        )

        helper.do_move(move_vertex, move_to);
      } else if (split_edge.valid()) {
        GEODE_DEBUG_ONLY(
          for (auto bf: mesh.faces(split_edge)) {
            if (bf.valid())
              minq_before = min(minq_before, quality[bf]);
          }
          std::cout << "    split " << split_edge << ": " << mesh.src(split_edge) << " -> " << mesh.dst(split_edge) << ", cost " << best_cost << std::endl;
        )
        changed_faces = helper.do_split(split_edge, move_to);
      } else {
        // can't remove this triangle, it's still there!
        still_needs_improvement.set(f);
        continue;
      }

      GEODE_DEBUG_ONLY(
        real minq_after = 1;
        for (auto nf: changed_faces) {
          minq_after = min(minq_after, nf.y);
        }
        std::cout << "    min quality " << minq_before << " -> " << minq_after << ", improvement " << minq_after - minq_before << std::endl;
      )

      Hashtable<VertexId> update_vertices;
      for (auto nf : changed_faces) {
        // update quality
        quality[nf.x] = nf.y;

        // remember vertices to update quadrics for
        for (auto v : mesh.vertices(nf.x)) {
          update_vertices.set(v);
        }

        if (nf.y < o.min_quality)
          still_needs_improvement.set(nf.x);
      }

      // update quadrics on update_vertices
      for (auto uv : update_vertices) {
        quadrics[uv] = compute_quadric(mesh, pos, uv);
        silhouette_quadrics[uv] = compute_silhouette_quadric(mesh, pos, uv, o.min_relevant_area);
      }

      improved_something = true;
    }

    GEODE_DEBUG_ONLY(real quality_after = mesh_quality(mesh, pos, Q));
    GEODE_DEBUG_ONLY(std::cout << "iteration " << iter << ": quality " << quality_before << " -> " << quality_after << " (diff = " << quality_after-quality_before << "), changed " << improved_something << std::endl);

    needs_improvement = still_needs_improvement;
    iter++;
  }

  return mesh_quality(mesh, pos, Q) >= o.min_quality;
}

// convenience versions of improve_mesh
template<class Quality, class EdgeLocked, class VertexLocked>
bool improve_mesh_inplace(MutableTriangleTopology &mesh, FieldId<Vector<real,3>, VertexId> &pos_id,
                          ImproveOptions const &o, FieldId<bool,HalfedgeId> edge_locked_id, FieldId<bool,VertexId> vertex_locked_id) {
  auto Q = [](Triangle<Vector<real,3>> const &t) { return t.quality(); };

  auto pos = mesh.field(pos_id);
  auto edge_locked = mesh.field(edge_locked_id);
  auto vertex_locked = mesh.field(vertex_locked_id);

  // translate edge locked field into set of locked vertex pairs
  Hashtable<Vector<VertexId,2>> locked_edge_pairs;
  for (auto h : mesh.halfedges()) {
    if (edge_locked[h]) {
      locked_edge_pairs.set(vec(mesh.src(h), mesh.dst(h)));
    }
  }
  auto EL = [&locked_edge_pairs](VertexId v0, VertexId v1) { return locked_edge_pairs.contains(vec(v0,v1)); };
  auto VL = [&vertex_locked](VertexId v) { return vertex_locked[v]; };

  return improve_mesh_inplace(mesh, pos, o, EL, VL, Q);
}

bool improve_mesh_inplace(MutableTriangleTopology &mesh,
                          FieldId<Vector<real,3>, VertexId> pos_id,
                          ImproveOptions const &o);

// positions are assumed to be in the default location and of the correct type.
Ref<MutableTriangleTopology> improve_mesh(MutableTriangleTopology const &mesh, real min_quality, real max_distance, real max_silhouette_distance, real min_normal_dot = .8, int max_iter = 20, bool can_change_topology = true, real min_relevant_area = 1e-12, real min_quality_improvement = 1e-6);

}
