//#####################################################################
// Evaluate signed distances between a point cloud and a mesh
//#####################################################################
#include <geode/geometry/surface_levelset.h>
#include <geode/geometry/ParticleTree.h>
#include <geode/geometry/SimplexTree.h>
#include <geode/geometry/Segment.h>
#include <geode/geometry/Triangle3d.h>
#include <geode/array/ProjectedArray.h>
#include <geode/array/ConstantMap.h>
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

static const bool profile = false;
GEODE_UNUSED static uint64_t evaluation_count;

static inline T lower_bound_sqr_phi(const TV& n1, const Box<TV>& n2) {
  return sqr_magnitude(n1-n2.clamp(n1));
}

static inline T lower_bound_sqr_phi(const Box<TV>& n1, const Box<TV>& n2) {
  return sqr_magnitude((n1-n2).clamp(TV()));
}

template<int d> struct Helper {
  const ParticleTree<TV>& particles;
  const SimplexTree<TV,d>& surface;
  RawArray<T> sqr_phi_node;
  RawArray<CloseInfo<d>> info; // phi = sqr_phi, normal = delta

  void eval(const int pn, const int sn) const {
    const Box<TV> &pbox = particles.boxes[pn],
                  &sbox = surface.boxes[sn];
    const bool pleaf = particles.is_leaf(pn),
               sleaf = surface.is_leaf(sn);
    if (pleaf && sleaf) { // Two leaves: compute all pairwise distances
      sqr_phi_node[pn] = 0;
      const auto particle_prims = particles.prims(pn);
      const auto surface_prims = surface.prims(sn);
      for (const int p : particle_prims) {
        if (info[p].phi > lower_bound_sqr_phi(particles.X[p],sbox))
          for (const int t : surface_prims) {
            if (profile)
              evaluation_count++;
            const auto close = surface.simplices[t].closest_point(particles.X[p]);
            const TV delta = particles.X[p] - close.x;
            const T sd = sqr_magnitude(delta);
            if (info[p].phi > sd)
              info[p] = CloseInfo<d>({sd,delta,t,close.y});
          }
        sqr_phi_node[pn] = max(sqr_phi_node[pn],info[p].phi);
      }
    } else if (pleaf || (!sleaf && pbox.sizes().max()<=sbox.sizes().max())) {
      // Recurse into surface_node
      const auto sc = surface.children(sn);
      const auto bounds = vec(lower_bound_sqr_phi(pbox,surface.boxes[sc.x]),
                              lower_bound_sqr_phi(pbox,surface.boxes[sc.y]));
      const int c = bounds.argmin();
      if (sqr_phi_node[pn] > bounds[c])
        eval(pn,sc[c]);
      if (sqr_phi_node[pn] > bounds[1-c])
        eval(pn,sc[1-c]);
    } else { // Recurse into particle_node
      sqr_phi_node[pn] = 0;
      for (const int c : particles.children(pn)) {
        if (sqr_phi_node[c] > lower_bound_sqr_phi(particles.boxes[c],sbox))
          eval(c,sn);
        sqr_phi_node[pn] = max(sqr_phi_node[pn],sqr_phi_node[c]);
      }
    }
  }
};
}

static TV normal_flip(const Segment<TV>& seg, const TV u) {
  const auto v = seg.vector();
  const T vv = sqr_magnitude(v);
  if (vv) {
    const auto n = u-dot(u,v)/sqr_magnitude(v)*v;
    const T nn = sqr_magnitude(n);
    if (nn)
      return n/sqrt(nn);
  }
  return normalized(u);
}
static TV normal_flip(const Triangle<TV>& tri, const TV u) {
  const TV& n = tri.n;
  return dot(u,n)>0 ? n : -n;
}
static TV normal_noflip(const Segment<TV>& seg) { GEODE_UNREACHABLE(); }
static TV normal_noflip(const Triangle<TV>& tri) { return tri.n; }

template<int d> void surface_levelset(const ParticleTree<TV>& particles, const SimplexTree<TV,d>& surface,
                                      RawArray<typename Hide<CloseInfo<d>>::type> info,
                                      const T max_distance, const bool compute_signs) {
  GEODE_ASSERT(particles.X.size()==info.size());
  const T sqr_max_distance = sqr(max_distance);
  for (auto& I : info) {
    I.phi = sqr_max_distance;
    I.simplex = -1;
  }
  if (profile)
    evaluation_count = 0;
  const auto sqr_phi_node = constant_map(particles.nodes(),sqr_max_distance).copy();
  if (particles.X.size() && surface.simplices.size())
    Helper<d>({particles,surface,sqr_phi_node,info}).eval(0,0);
  if (profile) {
    long slow_count = (long)particles.X.size()*surface.simplices.size();
    cout << "particles = "<<particles.X.size()<<", per particle "<<evaluation_count/particles.X.size()<<endl;
    cout << "simplices = "<<surface.simplices.size()<<", per simplex "<<evaluation_count/surface.simplices.size()<<endl;
    cout << "evaluation count = "<<evaluation_count<<" / "<<slow_count<<" = "<<(T)evaluation_count / slow_count<<endl;
  }
  const T epsilon = sqrt(numeric_limits<T>::epsilon())*max(particles.bounding_box().sizes().max(),
                                                             surface.bounding_box().sizes().max());
  if (d<TV::m-1 || !compute_signs)
    for (auto& I : info) {
      I.phi = sqrt(I.phi);
      I.normal = (I.simplex) < 0 ? TV() // Parentheses needed for gcc 4.9 bug
               : I.phi > epsilon ? I.normal / I.phi
                                 : normal_flip(surface.simplices[I.simplex],I.normal);
    }
  else // compute_signs
    for (const int i : range(info.size())) {
      auto& I = info[i];
      I.phi = sqrt(I.phi);
      if ((I.simplex) < 0) // Parentheses needed for gcc 4.9 bug
        I.normal = TV();
      else {
        try {
          const bool inside = surface.inside_given_closest_point(particles.X[i],I.simplex,I.weights);
          if (inside)
            I.phi = -I.phi;
          if (abs(I.phi) > epsilon)
            I.normal /= I.phi;
          else
            I.normal = normal_noflip(surface.simplices[I.simplex]);
        } catch (const ArithmeticError&) { // Inside test failed, assume zero
          I.phi = 0;
          I.normal = normal_noflip(surface.simplices[I.simplex]);
        }
      }
    }
}

template<int d> Tuple<Array<T>,Array<TV>,Array<int>,Array<typename SimplexTree<TV,d>::Weights>>
surface_levelset(const ParticleTree<TV>& particles, const SimplexTree<TV,d>& surface,
                 const T max_distance, const bool compute_signs) {
  Array<CloseInfo<d>> info(particles.X.size(),uninit);
  surface_levelset(particles,surface,info,max_distance,compute_signs);
  return tuple(info.template project<T,&CloseInfo<d>::phi>().copy(),
               info.template project<TV,&CloseInfo<d>::normal>().copy(),
               info.template project<int,&CloseInfo<d>::simplex>().copy(),
               info.template project<typename SimplexTree<TV,d>::Weights,&CloseInfo<d>::weights>().copy());
}

#define INSTANTIATE(d) \
  template void surface_levelset(const ParticleTree<TV>&,const SimplexTree<TV,d>&,RawArray<CloseInfo<d>>,T,bool); \
  template Tuple<Array<T>,Array<TV>,Array<int>,Array<typename SimplexTree<TV,d>::Weights>> \
    surface_levelset(const ParticleTree<TV>&,const SimplexTree<TV,d>&,const T,const bool);
INSTANTIATE(1)
INSTANTIATE(2)

// For testing purposes
static Tuple<Array<T>,Array<TV>,Array<int>,Array<TV>>
slow_surface_levelset(const ParticleTree<TV>& particles, const SimplexTree<TV,2>& surface) {
  Array<T> distances(particles.X.size(),uninit);
  Array<TV> directions(particles.X.size(),uninit);
  Array<int> simplices(particles.X.size(),uninit);
  Array<TV> weights(particles.X.size(),uninit);
  distances.fill(FLT_MAX);
  for (const int p : range(particles.X.size()))
    for (const int t : range(surface.simplices.size())) {
      const auto close = surface.simplices[t].closest_point(particles.X[p]);
      TV delta = particles.X[p]-close.x;
      T sqr_distance = sqr_magnitude(delta);
      if (distances[p] > sqr_distance) {
        distances[p] = sqr_distance;
        directions[p] = delta;
        simplices[p] = t;
        weights[p] = close.y;
      }
    }
  for (int i=0;i<distances.size();i++) {
    distances[i] = sqrt(distances[i]);
    directions[i] = distances[i] ? directions[i]/distances[i] : TV(1,0,0);
  }
  return tuple(distances,directions,simplices,weights);
}

}
using namespace geode;

void wrap_surface_levelset() {
  GEODE_FUNCTION_2(surface_levelset_c3d,static_cast<Tuple<Array<T>,Array<TV>,Array<int>,Array<T>>(*)(
    const ParticleTree<TV>&,const SimplexTree<TV,1>&,T,bool)>(surface_levelset))
  GEODE_FUNCTION_2(surface_levelset_s3d,static_cast<Tuple<Array<T>,Array<TV>,Array<int>,Array<TV>>(*)(
    const ParticleTree<TV>&,const SimplexTree<TV,2>&,T,bool)>(surface_levelset))
  GEODE_FUNCTION(slow_surface_levelset)
}
