//#####################################################################
// Evaluate signed distances between a point cloud and a mesh
//#####################################################################
#include <geode/geometry/surface_levelset.h>
#include <geode/geometry/ParticleTree.h>
#include <geode/geometry/SimplexTree.h>
#include <geode/geometry/Triangle3d.h>
#include <geode/array/ProjectedArray.h>
#include <geode/python/wrap.h>
#include <geode/utility/Log.h>
#include <limits>
namespace geode {

typedef real T;
typedef Vector<T,3> TV;
using Log::cout;
using std::endl;
using std::numeric_limits;

namespace {

const bool profile = false;
long evaluation_count;

inline T lower_bound_sqr_phi(const TV& n1, const Box<TV>& n2) {
  return sqr_magnitude(n1-n2.clamp(n1));
}

inline T lower_bound_sqr_phi(const Box<TV>& n1, const Box<TV>& n2) {
  return sqr_magnitude((n1-n2).clamp(TV()));
}

struct Helper {
  const ParticleTree<TV>& particles;
  const SimplexTree<TV,2>& surface;
  RawArray<T> sqr_phi_node;
  RawArray<CloseTriangleInfo> info; // phi = sqr_phi, normal = delta

  Helper(const ParticleTree<TV>& particles, const SimplexTree<TV,2>& surface, RawArray<T> sqr_phi_node, RawArray<CloseTriangleInfo> info)
    : particles(particles), surface(surface), sqr_phi_node(sqr_phi_node), info(info) {}

  void evaluate(const int particle_n, const int surface_n) const {
    const Box<TV> &particle_box = particles.boxes[particle_n],
                  &surface_box = surface.boxes[surface_n];
    const bool particle_leaf = particles.is_leaf(particle_n),
               surface_leaf = surface.is_leaf(surface_n);
    if (particle_leaf && surface_leaf) { // Two leaves: compute all pairwise distances
      sqr_phi_node[particle_n] = 0;
      RawArray<const int> particle_prims = particles.prims(particle_n);
      RawArray<const int> surface_prims = surface.prims(surface_n);
      for (int p : particle_prims) {
        if (info[p].phi > lower_bound_sqr_phi(particles.X[p],surface_box))
          for (int t : surface_prims) {
            if (profile)
              evaluation_count++;
            const auto close = surface.simplices[t].closest_point(particles.X[p]);
            TV delta = particles.X[p] - close.x;
            T sd = sqr_magnitude(delta);
            if (info[p].phi > sd) {
              info[p].phi = sd;
              info[p].normal = delta;
              info[p].triangle = t;
              info[p].weights = close.y;
            }
          }
        sqr_phi_node[particle_n] = max(sqr_phi_node[particle_n],info[p].phi);}
    } else if (particle_leaf || (!surface_leaf && particle_box.sizes().max()<=surface_box.sizes().max())) {
      // Recurse into surface_node
      int surface_ns[2];
      T bounds[2];
      for (int c=0;c<2;c++) {
        surface_ns[c] = surface.child(surface_n,c);
        bounds[c] = lower_bound_sqr_phi(particle_box,surface.boxes[surface_ns[c]]);
      }
      int c = bounds[0]<=bounds[1]?0:1;
      if (sqr_phi_node[particle_n] > bounds[c])
        evaluate(particle_n,surface_ns[c]);
      if (sqr_phi_node[particle_n] > bounds[1-c])
        evaluate(particle_n,surface_ns[1-c]);
    } else { // Recurse into particle_node
      sqr_phi_node[particle_n] = 0;
      for (int c=0;c<2;c++) {
        int pn = particles.child(particle_n,c);
        if (sqr_phi_node[pn] > lower_bound_sqr_phi(particles.boxes[pn],surface_box))
          evaluate(pn,surface_n);
        sqr_phi_node[particle_n] = max(sqr_phi_node[particle_n],sqr_phi_node[pn]);
      }
    }
  }
};

}

void evaluate_surface_levelset(const ParticleTree<TV>& particles, const SimplexTree<TV,2>& surface, RawArray<CloseTriangleInfo> info, T max_distance, bool compute_signs) {
  GEODE_ASSERT(particles.X.size()==info.size());
  const T sqr_max_distance = sqr(max_distance);
  for (int i=0;i<info.size();i++) {
    info[i].phi = sqr_max_distance;
    info[i].triangle = -1;
  }
  evaluation_count = 0;
  Array<T> sqr_phi_node(particles.nodes(),uninit);
  sqr_phi_node.fill(sqr_max_distance);
  if (particles.X.size() && surface.simplices.size())
    Helper(particles,surface,sqr_phi_node,info).evaluate(0,0);
  if (profile) {
    long slow_count = (long)particles.X.size()*surface.simplices.size();
    cout<<"particles = "<<particles.X.size()<<", per particle "<<evaluation_count/particles.X.size()<<endl;
    cout<<"triangles = "<<surface.simplices.size()<<", per triangle "<<evaluation_count/surface.simplices.size()<<endl;
    cout << "evaluation count = "<<evaluation_count<<" / "<<slow_count<<" = "<<(T)evaluation_count / slow_count<<endl;
  }
  T epsilon = sqrt(numeric_limits<T>::epsilon())*max(particles.bounding_box().sizes().max(),surface.bounding_box().sizes().max());
  if (!compute_signs)
    for (int i=0;i<info.size();i++) {
    CloseTriangleInfo& I = info[i];
    I.phi = sqrt(I.phi);
    if (I.triangle<0)
      I.normal = TV();
    else if (I.phi>epsilon)
      I.normal /= I.phi;
    else {
      const TV& n = surface.simplices[I.triangle].n;
      I.normal = dot(I.normal,n)>0?n:-n;
    }
  } else // compute_signs
    for (int i=0;i<info.size();i++) {
      CloseTriangleInfo& I = info[i];
      I.phi = sqrt(I.phi);
      if (I.triangle<0)
        I.normal = TV();
      else {
        try {
          const bool inside = surface.inside_given_closest_point(particles.X[i],I.triangle,I.weights);
          if (inside)
            I.phi = -I.phi;
          if (abs(I.phi)>epsilon)
            I.normal /= I.phi;
          else
            I.normal = surface.simplices[I.triangle].n;
        } catch (const ArithmeticError&) { // Inside test failed, assume zero
          I.phi = 0;
          I.normal = surface.simplices[I.triangle].n;
        }
      }
    }
}

Tuple<Array<T>,Array<TV>,Array<int>,Array<TV>> evaluate_surface_levelset(const ParticleTree<TV>& particles, const SimplexTree<TV,2>& surface, T max_distance, bool compute_signs) {
  Array<CloseTriangleInfo> info(particles.X.size(),uninit);
  evaluate_surface_levelset(particles,surface,info,max_distance,compute_signs);
  return tuple(info.project<T,&CloseTriangleInfo::phi>().copy(),
               info.project<TV,&CloseTriangleInfo::normal>().copy(),
               info.project<int,&CloseTriangleInfo::triangle>().copy(),
               info.project<TV,&CloseTriangleInfo::weights>().copy());
}

// For testing purposes
static Tuple<Array<T>,Array<TV>,Array<int>,Array<TV>> slow_evaluate_surface_levelset(const ParticleTree<TV>& particles,const SimplexTree<TV,2>& surface) {
  Array<T> distances(particles.X.size(),uninit);
  Array<TV> directions(particles.X.size(),uninit);
  Array<int> triangles(particles.X.size(),uninit);
  Array<TV> weights(particles.X.size(),uninit);
  distances.fill(FLT_MAX);
  for (int p=0;p<particles.X.size();p++)
    for (int t=0;t<surface.simplices.size();t++) {
      const auto close = surface.simplices[t].closest_point(particles.X[p]);
      TV delta = particles.X[p]-close.x;
      T sqr_distance = sqr_magnitude(delta);
      if (distances[p] > sqr_distance) {
        distances[p] = sqr_distance;
        directions[p] = delta;
        triangles[p] = t;
        weights[p] = close.y;
      }
    }
  for (int i=0;i<distances.size();i++) {
    distances[i] = sqrt(distances[i]);
    directions[i] = distances[i]?directions[i]/distances[i]:TV(1,0,0);
  }
  return tuple(distances,directions,triangles,weights);
}

}
using namespace geode;

void wrap_surface_levelset() {
  GEODE_FUNCTION_2(evaluate_surface_levelset,static_cast<Tuple<Array<T>,Array<TV>,Array<int>,Array<TV>>(*)(const ParticleTree<TV>&,const SimplexTree<TV,2>&,T,bool)>(evaluate_surface_levelset))
  GEODE_FUNCTION(slow_evaluate_surface_levelset)
}
