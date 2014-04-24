// Measure the mean and Gaussian curvatures of meshes

#include <geode/config.h>
#ifdef GEODE_OPENMESH

#include <geode/openmesh/curvature.h>
#include <geode/array/Field.h>
#include <geode/python/wrap.h>
#include <geode/utility/Log.h>
#include <geode/vector/normalize.h>
#include <geode/math/copysign.h>
namespace geode {

using Log::cout;
using std::endl;
typedef double T;
typedef Vector<T,3> TV;

// See Meyer et al., "Discrete differential geometry operators for triangulated 2-manifolds".

static inline T cot(const TV& e0, const TV& e1) {
  return dot(e0,e1)/magnitude(cross(e0,e1));
}

Field<T,VertexHandle> mean_curvatures(const TriMesh& mesh) {
  Field<T,VertexHandle> area(mesh.n_vertices()); // Actually 8*area
  Field<TV,VertexHandle> normal(mesh.n_vertices());
  Field<TV,VertexHandle> Hn(mesh.n_vertices()); // Actually 4*Hn
  for (const auto f : mesh.face_handles()) {
    const auto v = mesh.vertex_handles(f);
    const TV x0 = mesh.point(v.x),
             x1 = mesh.point(v.y),
             x2 = mesh.point(v.z),
             x01 = x1-x0,
             x12 = x2-x1,
             x20 = x0-x2;
    const T cot0 = cot(x01,x20),
            cot1 = cot(x01,x12),
            cot2 = cot(x12,x20);
    // Compute A_mixed as in Meyer et al.
    const TV n = mesh.calc_face_normal(f);
    if (cot0<=0 && cot1<=0 && cot2<=0) { // Voronoi case
      const T area0 = cot0*sqr_magnitude(x12),
              area1 = cot1*sqr_magnitude(x20),
              area2 = cot2*sqr_magnitude(x01);
      area[v.x] -= (area1+area2);
      area[v.y] -= (area2+area0);
      area[v.z] -= (area0+area1);
      normal[v.x] -= (area1+area2)*n;
      normal[v.y] -= (area2+area0)*n;
      normal[v.z] -= (area0+area1)*n;
    } else { // One of the triangles is obtuse
      const T a = 2*mesh.area(f);
      area[v.x] += (1+(cot0>0))*a;
      area[v.y] += (1+(cot1>0))*a;
      area[v.z] += (1+(cot2>0))*a;
      normal[v.x] += (1+(cot0>0))*a*n;
      normal[v.y] += (1+(cot1>0))*a*n;
      normal[v.z] += (1+(cot2>0))*a*n;
    }
    // Accumulate mean curvature normal
    Hn[v.x] += cot2*x01-cot1*x20;
    Hn[v.y] += cot0*x12-cot2*x01;
    Hn[v.z] += cot1*x20-cot0*x12;
  }
  Field<T,VertexHandle> H(mesh.n_vertices(),false);
  for (const auto v : mesh.vertex_handles()) {
    const TV n = normal[v];
    H[v] = 2/area[v] * (mesh.is_boundary(v) ? dot(Hn[v],normalized(n))
                                            : copysign(magnitude(Hn[v]),dot(Hn[v],n)));
  }
  return H;
}

Field<T,VertexHandle> gaussian_curvatures(const TriMesh& mesh) {
  // Compute mixed areas
  Field<T,VertexHandle> area(mesh.n_vertices()); // Actually 8*area
  for (const auto f : mesh.face_handles()) {
    const auto v = mesh.vertex_handles(f);
    const TV x0 = mesh.point(v.x),
             x1 = mesh.point(v.y),
             x2 = mesh.point(v.z),
             x01 = x1-x0,
             x12 = x2-x1,
             x20 = x0-x2;
    const T cot0 = cot(x01,x20),
            cot1 = cot(x01,x12),
            cot2 = cot(x12,x20);
    // Compute A_mixed as in Meyer et al.
    if (cot0<=0 && cot1<=0 && cot2<=0) { // Voronoi case
      const T area0 = cot0*sqr_magnitude(x12),
              area1 = cot1*sqr_magnitude(x20),
              area2 = cot2*sqr_magnitude(x01);
      area[v.x] -= (area1+area2);
      area[v.y] -= (area2+area0);
      area[v.z] -= (area0+area1);
    } else { // One of the triangles is obtuse
      const T a = 2*mesh.area(f);
      area[v.x] += (1+(cot0>0))*a;
      area[v.y] += (1+(cot1>0))*a;
      area[v.z] += (1+(cot2>0))*a;
    }
  }
  // Compute curvatures
  Field<T,VertexHandle> K(mesh.n_vertices(),false);
  for (const auto v : mesh.vertex_handles()) {
    const TV x = mesh.point(v);
    T sum = 0;
    for (auto e=mesh.cvoh_iter(v);e;++e)
      if (!mesh.is_boundary(e)) {
        const auto v0 = mesh.to_vertex_handle(e),
                   v1 = mesh.from_vertex_handle(mesh.prev_halfedge_handle(e)); 
        sum += angle_between(mesh.point(v0)-x,mesh.point(v1)-x);
      }
    K[v] = 8*((1+!mesh.is_boundary(v))*pi-sum)/area(v);
  }
  return K;
}

}
using namespace geode;

void wrap_curvature() {
  GEODE_FUNCTION(mean_curvatures)
  GEODE_FUNCTION(gaussian_curvatures)
}
#endif
