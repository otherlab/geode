#pragma once

#include <geode/mesh/TriangleTopology.h>
#include <geode/mesh/quadric.h>
#include <geode/math/lerp.h>
#include <geode/solver/brent.h>

namespace geode {

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
struct Allowed {
  MutableTriangleTopology const &mesh;
  Field<Vector<real,3>, VertexId> const &pos;
  Field<real, FaceId> const &quality;
  Field<Quadric, VertexId> const &quadrics;
  Quality &Q;
  EdgeLocked &EL;
  VertexLocked &VL;
  real min_normal_dot;
  real max_distance;
  real min_quality_improvement;

  Allowed(MutableTriangleTopology const &mesh, Field<Vector<real,3>, VertexId> const &pos,
          Field<real, FaceId> const &quality, Field<Quadric, VertexId> const &quadrics,
          Quality &Q, EdgeLocked &EL, VertexLocked &VL,
          real min_normal_dot, real max_distance, real min_quality)
  : mesh(mesh)
  , pos(pos)
  , quality(quality)
  , quadrics(quadrics)
  , Q(Q)
  , EL(EL)
  , VL(VL)
  , min_normal_dot(min_normal_dot)
  , max_distance(max_distance)
  , min_quality_improvement(1e-6)
  {}

  // costs are in [0,1), or infinity
  real quadric_cost(real quadric_error) {
    if (quadric_error >= max_distance*max_distance)
      return numeric_limits<real>::infinity();
    return quadric_error/(max_distance*max_distance);
  }
  real normal_cost(real normal_dot) {
    if (normal_dot <= min_normal_dot)
      return numeric_limits<real>::infinity();
    return lerp(normal_dot, min_normal_dot, 1., 1., 0.);
  }

  // criteria:
  // - cannot move vertices further than allowed by quadrics
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
    FaceId f1 = mesh.face(h);
    FaceId f2 = mesh.face(mesh.reverse(h));
    auto old_t1 = mesh.triangle(pos, f1);
    auto old_t2 = mesh.triangle(pos, f2);
    real old_q1 = quality[f1];
    real old_q2 = quality[f2];
    real old_a1 = old_t1.area();
    real old_a2 = old_t2.area();
    auto old_n1 = old_t1.normal();
    auto old_n2 = old_t2.normal();

    // new faces
    auto new_t1 = Triangle<Vector<real,3>>(old_t1.x2, old_t2.x2, old_t1.x1);
    auto new_t2 = Triangle<Vector<real,3>>(old_t2.x2, old_t1.x2, old_t2.x1);
    real new_q1 = Q(new_t1);
    real new_q2 = Q(new_t2);
    real new_a1 = new_t1.area();
    real new_a2 = new_t2.area();
    auto new_n1 = new_t1.normal();
    auto new_n2 = new_t2.normal();

    if (min_quality_improvement+min(old_q1, old_q2) >= min(new_q1, new_q2)) {
      GEODE_DEBUG_ONLY(std::cout << "    illegal flip for " << mesh.src(h) << " - " << mesh.dst(h) << ", q " << min(old_q1, old_q2) << " -> " << min(new_q1, new_q2) << ", improvement " << min(new_q1, new_q2)-min(old_q1, old_q2) << std::endl);
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>());
    }

    // compute minimum normal dot product between any pair of old and new faces
    // for which a normal can be computed
    real min_dot = 1;

    if (old_a1 && new_a1)
      min_dot = min(min_dot, dot(old_n1, new_n1));
    if (old_a2 && new_a1)
      min_dot = min(min_dot, dot(old_n2, new_n1));
    if (old_a1 && new_a2)
      min_dot = min(min_dot, dot(old_n1, new_n2));
    if (old_a2 && new_a2)
      min_dot = min(min_dot, dot(old_n2, new_n2));

    Array<Tuple<FaceId,real>> changed;
    changed.append(tuple(f1, new_q1));
    changed.append(tuple(f2, new_q2));

    GEODE_DEBUG_ONLY(std::cout << "    checking flip for " << mesh.src(h) << " - " << mesh.dst(h) << ", dot: " << min_normal_dot << " (areas " << old_a1 << ", " << old_a2 << " -> " << new_a1 << ", " << new_a2 << "), q " << min(old_q1, old_q2) << " -> " << min(new_q1, new_q2) << ", improvement " << min(new_q1, new_q2)-min(old_q1, old_q2) << std::endl);

    return tuple(normal_cost(min_dot), changed);
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

    if (!mesh.is_collapse_safe(h))
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>());

    // compute quadric cost
    real quadric_cost = this->quadric_cost(quadrics[v0](pos[v1]));
    if (quadric_cost == numeric_limits<real>::infinity()) {
      GEODE_DEBUG_ONLY(std::cout << "    illegal collapse " << mesh.src(h) << " -> " << mesh.dst(h) << ", bad quadric: " << quadrics[v0](pos[v1]) << std::endl);
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>());
    }

    Array<Tuple<FaceId,real>> new_faces;
    real minq_before = 1;
    real max_normal_cost = 0;

    for (auto f : mesh.incident_faces(v0)) {
      // compute minimum quality before
      real old_q = quality[f];
      minq_before = min(minq_before, old_q);

      if (mesh.vertices(f).contains(v1))
        continue;

      // get current normal
      auto t = mesh.triangle(pos, f);
      real old_a = t.area();
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
      auto new_n = new_t.normal();

      // compute whether quality allows this operation
      if (new_q < minq_before) {
        GEODE_DEBUG_ONLY(std::cout << "    illegal collapse " << mesh.src(h) << " -> " << mesh.dst(h) << ", quadric: " << quadrics[v0](pos[v1]) << ", quality decreases" << std::endl);
        return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>());
      }

      // compute normal cost
      if (old_a && new_a) {
        max_normal_cost = max(max_normal_cost, normal_cost(dot(old_n, new_n)));

        assert(max_normal_cost >= 0);

        if (max_normal_cost == numeric_limits<real>::infinity()) {
          GEODE_DEBUG_ONLY(std::cout << "    illegal collapse " << mesh.src(h) << " -> " << mesh.dst(h) << ", quadric: " << quadrics[v0](pos[v1]) << ", bad normal dot " << dot(old_n, new_n) << ", areas " << old_a << ", " << new_a << std::endl);
          return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>());
        }
      }

      // remember modified faces and new qualities
      new_faces.append(tuple(f, new_q));
    }

    GEODE_DEBUG_ONLY(std::cout << "    checking collapse " << mesh.src(h) << " -> " << mesh.dst(h) << ", quadric: " << quadrics[v0](pos[v1]) << ", normal_cost " << max_normal_cost << std::endl);

    return tuple(quadric_cost + max_normal_cost, new_faces);
  }

  // this function additionally returns the optimal (while still allowed) move to position
  Tuple<real,Array<Tuple<FaceId,real>>, Vector<real,3>> check_move(VertexId v) {
    bool edge_locked = false;
    for (auto h : mesh.outgoing(v)) {
      edge_locked = edge_locked || EL(mesh.src(h), mesh.dst(h)) || EL(mesh.dst(h), mesh.src(h));
    }

    if (edge_locked || VL(v))
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>(), Vector<real,3>());

    // TODO: we should actually put a proper optimization here: optimize the minimum
    // incident triangle quality, subject quadric constraints on the vertex and
    // normal constraints on the incident faces.

    // For now, just guess that the (unweighted) centroid of the surrounding vertices,
    // projected onto the tangent plane of the vertex is a good target, and we use
    // brent to optimize only along that line
    int nvertices = 0;
    Vector<real,3> target;
    for (auto v2 : mesh.vertex_one_ring(v)) {
      target += pos[v2];
      nvertices++;
    }
    target /= nvertices;
    target = target.projected_orthogonal_to_unit_direction(mesh.normal(pos, v));

    auto faces = mesh.incident_faces(v);
    real before_min_q = 1;
    Array<real> areas;
    Array<Vector<real,3>> normals;
    for (auto f : faces) {
      before_min_q = min(before_min_q, quality[f]);
      auto t = mesh.triangle(pos,f);
      areas.append(t.area());
      normals.append(t.normal());
    }
    auto objective = [&](real x) {
      // minimize (1 - min_quality), or >1 if any of the normals move
      // too much, or if the quadric error is too high
      real min_q = 1;
      int i = 0;
      for (auto f : faces) {
        // get the triangle with changed v -> (1-x) * pos[v] + x * target
        auto p = (1-x)*pos[v] + x * target;
        auto verts = mesh.vertices(f);
        int k = verts.find(v);
        Triangle<Vector<real,3>> t(k == 0 ? p : pos[verts[0]],
                                   k == 1 ? p : pos[verts[1]],
                                   k == 2 ? p : pos[verts[2]]);

        // check quadrics
        if (quadrics[v](p) > max_distance*max_distance)
          return 1. + quadrics[v](p) - max_distance*max_distance; // make the objective smooth

        // compute quality
        min_q = min(min_q, Q(t));

        // check normal
        auto a = t.area();
        auto n = t.normal();

        auto old_a = areas[i];
        auto old_n = normals[i];

        if (old_a && a)
          if (dot(n,old_n) < min_normal_dot)
            return 1. + min_normal_dot - dot(n,old_n); // make the objective smooth

        i++;
      }
      return 1 - min_q;
    };

    auto res = brent(objective, vec(0.,1.), .01, 10);
    auto qafter = 1-res.y;
    auto p = (1-res.x)*pos[v] + res.x * target;

    if (qafter < before_min_q+min_quality_improvement) { // illegal or no (tangible) improvement?
      GEODE_DEBUG_ONLY(std::cout << "    illegal move: p-x " << p-pos[v] << ", q " << before_min_q << " -> " << qafter << ", improvement " << qafter-before_min_q << std::endl);
      return tuple(numeric_limits<real>::infinity(), Array<Tuple<FaceId,real>>(), Vector<real,3>());
    }

    Array<Tuple<FaceId,real>> new_faces;
    real min_dot = 1;
    GEODE_DEBUG_ONLY(real minq = 1);
    int i = 0;
    for (auto f : faces) {
      auto verts = mesh.vertices(f);
      int k = verts.find(v);
      Triangle<Vector<real,3>> t(k == 0 ? p : pos[verts[0]],
                                 k == 1 ? p : pos[verts[1]],
                                 k == 2 ? p : pos[verts[2]]);
      new_faces.append(tuple(f, Q(t)));

      // check normal
      auto a = t.area();
      auto n = t.normal();

      auto old_a = areas[i];
      auto old_n = normals[i];

      if (old_a && a)
        min_dot = min(min_dot, dot(n,old_n));

      i++;

      GEODE_DEBUG_ONLY(minq = min(minq, new_faces.back().y));
    }

    GEODE_DEBUG_ONLY(GEODE_ASSERT(fabs(minq - qafter) < 1e-12));
    GEODE_DEBUG_ONLY(std::cout << "    checking move: p-x " << p-pos[v] << ", cost " << quadrics[v](p) + normal_cost(min_dot) << ", q " << before_min_q << " -> " << minq << ", improvement " << minq-before_min_q << std::endl);

    return tuple(quadrics[v](p) + normal_cost(min_dot), new_faces, p);
  }
};

template<class Quality, class EdgeLocked, class VertexLocked>
bool improve_mesh_inplace(MutableTriangleTopology &mesh, Field<Vector<real,3>, VertexId> const &pos,
                          real min_quality, real max_distance, real min_normal_dot, int max_iter,
                          EdgeLocked &EL, VertexLocked &VL, Quality &Q) {

  GEODE_DEBUG_ONLY(std::cout.precision(18));

  // cache face qualities
  Field<real,FaceId> quality = mesh.create_compatible_face_field<real>();
  real old_mesh_quality = 1;
  Hashtable<FaceId> needs_improvement;
  for (auto f : mesh.faces()) {
    quality[f] = Q(mesh.triangle(pos, f));
    if (quality[f] < min_quality) {
      old_mesh_quality = min(old_mesh_quality, quality[f]);
      needs_improvement.set(f);
    }
  }

  // cache quadrics
  Field<Quadric,VertexId> quadrics = mesh.create_compatible_vertex_field<Quadric>();
  for (auto v : mesh.vertices()) {
    quadrics[v] = compute_quadric(mesh, pos, v);
  }

  Allowed<Quality, EdgeLocked, VertexLocked> allowed(mesh, pos, quality, quadrics, Q, EL, VL, min_normal_dot, max_distance, min_quality);

  bool improved_something = !needs_improvement.empty();
  int iter = 0;
  while (improved_something && iter < max_iter) {

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
      if (q >= min_quality)
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
        auto r = allowed.check_flip(he);
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
          auto r = allowed.check_collapse(he);
          if (r.x < best_cost) {
            flip_edge = HalfedgeId();
            best_cost = r.x;
            collapse_edge = he;
            changed_faces = r.y;
          }
        }
      }

      // check vertex moves only if nothing else works (they're expensive)
      VertexId move_vertex;
      Vector<real,3> move_to;
      if (!flip_edge.valid() && !collapse_edge.valid()) {
        for (auto v : mesh.vertices(f)) {
          auto r = allowed.check_move(v);
          if (r.x < best_cost) {
            collapse_edge = flip_edge = HalfedgeId();
            best_cost = r.x;
            move_vertex = v;
            move_to = r.z;
            changed_faces = r.y;
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

        flip_edge = mesh.flip_edge(flip_edge);
      } else if (collapse_edge.valid()) {
        GEODE_DEBUG_ONLY(
          for (auto bf: mesh.incident_faces(mesh.src(collapse_edge))) {
            minq_before = min(minq_before, quality[bf]);
          }
          std::cout << "    collapse " << collapse_edge << ", cost " << best_cost << std::endl;
        )

        mesh.collapse(collapse_edge);
      } else if (move_vertex.valid()) {
        GEODE_DEBUG_ONLY(
          for (auto bf: mesh.incident_faces(move_vertex)) {
            minq_before = min(minq_before, quality[bf]);
          }
          std::cout << "    move " << move_vertex << ", cost " << best_cost << std::endl;
        )

        pos[move_vertex] = move_to;
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

        if (nf.y < min_quality)
          still_needs_improvement.set(nf.x);
      }

      // update quadrics on update_vertices
      for (auto uv : update_vertices) {
        quadrics[uv] = compute_quadric(mesh, pos, uv);
      }

      improved_something = true;
    }

    GEODE_DEBUG_ONLY(real quality_after = mesh_quality(mesh, pos, Q));
    GEODE_DEBUG_ONLY(std::cout << "iteration " << iter << ": quality " << quality_before << " -> " << quality_after << " (diff = " << quality_after-quality_before << "), changed " << improved_something << std::endl);

    needs_improvement = still_needs_improvement;
    iter++;
  }

  return mesh_quality(mesh, pos, Q) >= min_quality;
}

// convenience versions of improve_mesh
template<class Quality, class EdgeLocked, class VertexLocked>
bool improve_mesh_inplace(MutableTriangleTopology &mesh, Field<Vector<real,3>, VertexId> const &pos,
                          real min_quality, real max_distance, real min_normal_dot, int max_iter,
                          Field<bool,HalfedgeId> const &edge_locked, Field<bool,VertexId> const &vertex_locked) {
  auto Q = [](Triangle<Vector<real,3>> const &t) { return t.quality(); };

  // translate edge locked field into set of locked vertex pairs
  Hashtable<Vector<VertexId,2>> locked_edge_pairs;
  for (auto h : mesh.halfedges()) {
    if (edge_locked[h]) {
      locked_edge_pairs.set(vec(mesh.src(h), mesh.dst(h)));
    }
  }
  auto EL = [&locked_edge_pairs](VertexId v0, VertexId v1) { return locked_edge_pairs.contains(vec(v0,v1)); };
  auto VL = [&vertex_locked](VertexId v) { return vertex_locked[v]; };

  return improve_mesh_inplace(mesh, pos, min_quality, max_distance, min_normal_dot, max_iter, EL, VL, Q);
}

bool improve_mesh_inplace(MutableTriangleTopology &mesh, Field<Vector<real,3>, VertexId> const &pos,
                          real min_quality, real max_distance, real min_normal_dot = .8, int max_iter = 20);

// positions are assumed to be in the default location and of the correct type.
Ref<MutableTriangleTopology> improve_mesh(MutableTriangleTopology const &mesh, real min_quality, real max_distance, real min_normal_dot = .8, int max_iter = 20);

}
