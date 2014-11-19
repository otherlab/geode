// Quadric-based mesh decimation

#include <geode/mesh/decimate.h>
#include <geode/python/wrap.h>
#include <geode/structure/Heap.h>
#include <geode/mesh/quadric.h>

namespace geode {

typedef real T;
typedef Vector<T,3> TV;

namespace {
// Binary heap of potential collapses
struct Heap : public HeapBase<Heap>, public Noncopyable {
  typedef HeapBase<Heap> Base;
  Array<Tuple<VertexId,T,VertexId>> heap; // src,badness,dst
  Field<int,VertexId> inv_heap;

  Heap(const int nv)
    : inv_heap(nv,uninit) {
    inv_heap.flat.fill(-1);
  }

  int size() const {
    return heap.size();
  }

  bool first(const int i, const int j) const {
    return heap[i].y <= heap[j].y;
  }

  void swap(const int i, const int j) {
    std::swap(heap[i],heap[j]);
    inv_heap[heap[i].x] = i;
    inv_heap[heap[j].x] = j;
  }

  Vector<VertexId,2> pop() {
    const auto e = heap[0];
    inv_heap[e.x] = -1;
    const auto p = heap.pop();
    if (size()) {
      heap[0] = p;
      inv_heap[heap[0].x] = 0;
      Base::move_downward(0);
    }
    return vec(e.x,e.z);
  }

  void set(const VertexId v, const T q, const VertexId dst) {
    const auto entry = tuple(v,q,dst);
    int i = inv_heap[v];
    if (i < 0)
      i = heap.append(entry);
    else
      heap[i] = entry;
    Base::move_up_or_down(i);
  }

  void erase(const VertexId v) {
    int& i = inv_heap[v];
    if (i >= 0) {
      const auto p = heap.pop();
      if (i < size()) {
        heap[i] = p;
        inv_heap[p.x] = i;
        Base::move_up_or_down(i);
      }
      i = -1;
    }
  }
};
}

void decimate_inplace(MutableTriangleTopology& mesh, RawField<TV,VertexId> X,
                      const T distance, const T max_angle, const int min_vertices, const T boundary_distance) {
  if (mesh.n_vertices() <= min_vertices)
    return;
  const T area = sqr(distance);
  const T sign_sqr_min_cos = sign_sqr(max_angle > .99*pi ? -1 : cos(max_angle));

  // Finds the best edge to collapse v along.  Returns (q(e),dst(e)).
  const auto best_collapse = [&mesh,X](const VertexId v) {
    Quadric q = compute_quadric(mesh,X,v);

    // Find the best edge, ignoring normal constraints
    T min_q = inf;
    HalfedgeId min_e;
    for (const auto e : mesh.outgoing(v)) {
      const T qx = q(X[mesh.dst(e)]);
      if (min_q > qx) {
        min_q = qx;
        min_e = e;
      }
    }
    return tuple(min_q,mesh.dst(min_e));
  };

  // Initialize quadrics and heap
  Heap heap(mesh.n_vertices_);
  for (const auto v : mesh.vertices()) {
    const auto qe = best_collapse(v);
    if (qe.x <= area)
      heap.inv_heap[v] = heap.heap.append(tuple(v,qe.x,qe.y));
  }
  heap.make();

  // Update the quadric information for a vertex
  const auto update = [&heap,best_collapse,area](const VertexId v) {
    const auto qe = best_collapse(v);
    if (qe.x <= area)
      heap.set(v,qe.x,qe.y);
    else
      heap.erase(v);
  };

  // Repeatedly collapse the best vertex
  while (heap.size()) {
    const auto v = heap.pop();

    // Do these vertices still exist?
    if (mesh.valid(v.x) && mesh.valid(v.y)) {
      const auto e = mesh.halfedge(v.x,v.y);

      // Is the collapse invalid?
      if (e.valid() && mesh.is_collapse_safe(e)) {
        const auto vs = mesh.src(e),
                   vd = mesh.dst(e);
        const auto xs = X[vs],
                   xd = X[vd];

        // Are we moving a boundary vertex too far from its two boundary lines?
        {
          const auto b = mesh.halfedge(vs);
          if (mesh.is_boundary(b)) {
            const auto x0 = X[mesh.dst(b)],
                       x1 = X[mesh.src(mesh.prev(b))];
            if (   line_point_distance(simplex(xs,x0),xd) > boundary_distance
                || line_point_distance(simplex(xs,x1),xd) > boundary_distance)
              goto bad;
          }
        }

        // Do the normals change too much?
        if (sign_sqr_min_cos > -1)
          for (const auto ee : mesh.outgoing(vs))
            if (e!=ee && !mesh.is_boundary(ee)) {
              const auto v2 = mesh.opposite(ee);
              if (v2 != vd) {
                const auto x1 = X[mesh.dst(ee)],
                           x2 = X[v2];
                const auto n0 = cross(x2-x1,xs-x1),
                           n1 = cross(x2-x1,xd-x1);
                if (sign_sqr(dot(n0,n1)) < sign_sqr_min_cos*sqr_magnitude(n0)*sqr_magnitude(n1))
                  goto bad;
              }
            }

        // Collapse vs onto vd, then update the heap
        mesh.unsafe_collapse(e);
        if (mesh.n_vertices() <= min_vertices)
          break;
        update(vd);
        for (const auto e : mesh.outgoing(vd))
          update(mesh.dst(e));
      }
    }
    bad:;
  }
}

Tuple<Ref<const TriangleTopology>,Field<const TV,VertexId>>
decimate(const TriangleTopology& mesh, RawField<const TV,VertexId> X,
         const T distance, const T max_angle, const int min_vertices, const T boundary_distance) {
  const auto rmesh = mesh.mutate();
  const auto rX = X.copy();
  decimate_inplace(rmesh,rX,distance,max_angle,min_vertices,boundary_distance);
  return Tuple<Ref<const TriangleTopology>,Field<const TV,VertexId>>(rmesh,rX);
}

}
using namespace geode;

void wrap_decimate() {
  GEODE_FUNCTION(decimate)
  GEODE_FUNCTION(decimate_inplace)
}
