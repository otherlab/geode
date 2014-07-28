// Segments

#include <geode/geometry/Segment.h>
#include <geode/geometry/Ray.h>
#include <geode/array/Array.h>
#include <geode/math/clamp.h>
#include <geode/math/givens.h>
#include <geode/math/copysign.h>
#include <geode/python/wrap.h>
#include <geode/random/Random.h>
#include <geode/structure/Tuple.h>
#include <geode/vector/Vector.h>
#include <geode/utility/Log.h>
namespace geode {

typedef real T;
typedef Vector<real,2> TV2;
typedef Vector<real,3> TV3;
using std::cout;
using std::endl;

#ifdef GEODE_PYTHON
template<class TV> PyObject* to_python(const Segment<TV>& seg) {
  return to_python(tuple(seg.x0,seg.x1));
}
template PyObject* to_python(const Segment<Vector<real,2>>&);
template PyObject* to_python(const Segment<Vector<real,3>>&);
#endif

template<class TV> real interpolation_fraction(const Segment<TV>& s, const TV p) {
  const TV v = s.x1-s.x0;
  const T vv = sqr_magnitude(v);
  return vv ? dot(p-s.x0,v)/vv : 0;
}

template<class TV> real clamped_interpolation_fraction(const Segment<TV>& s, const TV p) {
  const TV v = s.x1-s.x0;
  const T d = dot(p-s.x0,v);
  if (d <= 0)
    return 0;
  const T vv = sqr_magnitude(v);
  if (d >= vv)
    return 1;
  return d/vv;
}

// Use Givens rotations to change s to (0,0)-(x,0) with x >= 0.
static inline Vector<Vector<T,2>,1> standardize_segment_point(Segment<TV2>& s, TV2& p) {
  // Translate s.x0 to origin
  s.x1 -= s.x0;
  p    -= s.x0;
  s.x0 = TV2();

  // Zero s.x1.y with a Givens rotation
  Vector<Vector<T,2>,1> g;
  g.x = givens_and_apply(s.x1.x,s.x1.y);
  givens_apply(g.x,p.x,p.y);
  return g;
}

// Use Givens rotations to change s to (0,0,0)-(x,0,0) with x >= 0.
static inline Vector<Vector<T,2>,2> standardize_segment_point(Segment<TV3>& s, TV3& p) {
  // Translate s.x0 to origin
  s.x1 -= s.x0;
  p    -= s.x0;
  s.x0 = TV3();

  // Zero s.x1.yz with Givens rotations
  Vector<Vector<T,2>,2> g;
  g.x = givens_and_apply(s.x1.y,s.x1.z);
  givens_apply(g.x,p.y,p.z);
  g.y = givens_and_apply(s.x1.x,s.x1.y);
  givens_apply(g.y,p.x,p.y);
  return g;
}

template<int d> Tuple<T,Vector<T,d>,T> segment_point_distance_and_normal(Segment<Vector<T,d>> s, Vector<T,d> p) {
  typedef Vector<T,d> TV;
  const auto g = standardize_segment_point(s,p);
  T dist;
  TV n;
  T w;
  if (p.x < 0) {
    n = p;
    dist = magnitude(n);
    n /= dist;
    w = 0;
  } else if (p.x > s.x1.x) {
    n = p-s.x1;
    dist = magnitude(n);
    n /= dist;
    w = 1;
  } else {
    if (d==2) {
      dist = abs(p.y);
      n.y = copysign(1.,p.y);
    } else {
      dist = sqrt(sqr(p.y)+sqr(p[2]));
      if (dist) {
        n.y = p.y/dist;
        n[2] = p[2]/dist;
      } else {
        n.y = 1;
        n[2] = 0;
      }
    }
    w = s.x1.x ? p.x/s.x1.x : 0;
  }
  if (d==2)
    givens_unapply(g.x,n.x,n.y);
  else {
    givens_unapply(g[1],n.x,n.y);
    givens_unapply(g.x,n.y,n[2]);
  }
  return tuple(dist,n,w);
}

template<int d> Tuple<Vector<T,d>,T> segment_closest_point(const Segment<Vector<T,d>> s, const Vector<T,d> p) {
  const auto u = p-s.x0,
             v = s.x1-s.x0;
  const T uv = dot(u,v);
  if (uv <= 0)
    return tuple(s.x0,0.);
  const T vv = sqr_magnitude(v);
  if (uv >= vv)
    return tuple(s.x1,1.);
  const T w = uv/vv;
  return tuple(s.x0+w*v,w);
}

template<int d> T segment_point_sqr_distance(Segment<Vector<T,d>> s, Vector<T,d> p) {
  const auto u = p-s.x0,
             v = s.x1-s.x0;
  const T uv = dot(u,v);
  if (uv <= 0)
    return sqr_magnitude(u);
  const T vv = sqr_magnitude(v);
  if (uv >= vv)
    return sqr_magnitude(p-s.x1);
  // sqrt(uu-sqr(uv)/vv) would be faster, but is less accurate for extremely small distances
  return sqr_magnitude(u-uv/vv*v);
}

template<int d> T segment_point_distance(Segment<Vector<T,d>> s, Vector<T,d> p) {
  return sqrt(segment_point_sqr_distance(s,p));
}

Tuple<T,TV3,T> line_point_distance_and_normal(Segment<TV3> s, TV3 p) {
  const auto g = standardize_segment_point(s,p);
  auto u = p.yz();
  auto dist = normalize(u);
  TV3 n(0,u.x,u.y);
  T w = s.x1.x?p.x/s.x1.x:0;
  givens_unapply(g.y,n.x,n.y);
  givens_unapply(g.x,n.y,n.z);
  return tuple(dist,n,w);
}

template<int d> Tuple<Vector<T,d>,T> line_closest_point(Segment<Vector<T,d>> s, Vector<T,d> p) {
  const auto v = s.x1-s.x0;
  const T vv = sqr_magnitude(v);
  if (!vv)
    return tuple(s.x0,T(0)); // x0 and x1 are a single point
  const T w = dot(p-s.x0,v)/vv;
  return tuple(s.x0+w*v,w);
}

template<int d> T line_point_distance(Segment<Vector<T,d>> s, Vector<T,d> p) {
  return magnitude(line_closest_point(s,p).x-p);
}

// Use Givens rotations to change first segment to (0,0)-(x,0) with x >= 0.
static inline Vector<Vector<T,2>,1> standardize_segments(Segment<Vector<T,2>>& s0, Segment<Vector<T,2>>& s1) {
  // Translate s0.x0 to origin
  s0.x1 -= s0.x0;
  s1.x0 -= s0.x0;
  s1.x1 -= s0.x0;
  s0.x0 = Vector<T,2>();

  // Zero s0.x1.y with an xy Givens transform
  const auto g = givens_and_apply(s0.x1.x,s0.x1.y);
  givens_apply(g,s1.x0.x,s1.x0.y);
  givens_apply(g,s1.x1.x,s1.x1.y);
  return vec(g);
}

// Use Givens rotations to change segments to (0,0,0)-(x,0,0) and (p,z)-(q,z) with x >= 0.
static inline Vector<Vector<T,2>,3> standardize_segments(Segment<Vector<T,3>>& s0, Segment<Vector<T,3>>& s1) {
  // Translate s0.x0 to origin
  s0.x1 -= s0.x0;
  s1.x0 -= s0.x0;
  s1.x1 -= s0.x0;
  s0.x0 = Vector<T,3>();

  // Zero s0.x1.z with a yz Givens transform
  Vector<Vector<T,2>,3> g;
  g.x = givens_and_apply(s0.x1.y,s0.x1.z);
  givens_apply(g.x,s1.x0.y,s1.x0.z);
  givens_apply(g.x,s1.x1.y,s1.x1.z);

  // Zero s0.x1.y with an xy Givens transform
  g.y = givens_and_apply(s0.x1.x,s0.x1.y);
  givens_apply(g.y,s1.x0.x,s1.x0.y);
  givens_apply(g.y,s1.x1.x,s1.x1.y);

  // Make s1.x0.z==s1.x1.z with a yz Givens transform
  auto d = s1.x1.yz()-s1.x0.yz();
  g.z = givens_and_apply(d.x,d.y);
  givens_apply(g.z,s1.x0.y,s1.x0.z);
  s1.x1.y = g.z.x*s1.x1.y+g.z.y*s1.x1.z;
  s1.x1.z = s1.x0.z;

  return g;
}

Tuple<T,TV3,Vector<T,2>> line_line_distance_and_normal(Segment<TV3> s0, Segment<TV3> s1) {
  const auto g = standardize_segments(s0,s1);

  // Compute weights
  const T ndy = s1.x0.y-s1.x1.y;
  const T w1 = ndy?s1.x0.y/ndy:0;
  const T w0 = s0.x1.x?(s1.x0.x+w1*(s1.x1.x-s1.x0.x))/s0.x1.x:0;

  // Compute shortest vector
  TV3 n(0,0,copysign(1,s1.x0.z));
  givens_unapply(g.z,n.y,n.z);
  givens_unapply(g.y,n.x,n.y);
  givens_unapply(g.x,n.y,n.z);
  return tuple(abs(s1.x0.z),n,vec(w0,w1));
}

template<int d> T segment_segment_distance(Segment<Vector<T,d>> s0, Segment<Vector<T,d>> s1) {
  standardize_segments(s0,s1);
  // If the global minimum occurs in the interior, use that
  if (s0.x1.x) {
    const T ndy = s1.x0.y-s1.x1.y;
    if (ndy) {
      const T w1 = s1.x0.y/ndy;
      if (0<=w1 && w1<=1) {
        const T w0 = s1.x0.x+w1*(s1.x1.x-s1.x0.x);
        if (0<=w0 && w0<=s0.x1.x)
          return d==2 ? 0 : abs(s1.x0[2]);
      }
    }
  }
  // Otherwise, minimize over all boundary cases.  This can be further optimized a fair bit, but for now we go for simplicity.
  return min(segment_point_distance(s0,s1.x0),
             segment_point_distance(s0,s1.x1),
             segment_point_distance(s1,s0.x0),
             segment_point_distance(s1,s0.x1));
}

template<int d> Tuple<T,TV3,TV2> segment_segment_distance_and_normal(const Segment<Vector<T,d>> s0, const Segment<Vector<T,d>> s1) {
  {
    auto s0_ = s0,
         s1_ = s1;
    const auto g = standardize_segments(s0_,s1_);
    const auto x0 = s1_.x0, x1 = s1_.x1;
    // If the global minimum occurs in the interior, use that
    if (s0_.x1.x) {
      const T ndy = x0.y-x1.y;
      if (ndy) {
        const T w1 = x0.y/ndy;
        if (0<=w1 && w1<=1) {
          const T w0 = (x0.x+w1*(x1.x-x0.x))/s0_.x1.x;
          if (0<=w0 && w0<=1) {
            if (d==2)
              return tuple(0.,TV3(0,0,1),vec(w0,w1));
            else {
              TV3 n(0,0,copysign(1,x0[2]));
              givens_unapply(g[2],n.y,n.z);
              givens_unapply(g[1],n.x,n.y);
              givens_unapply(g[0],n.y,n.z);
              return tuple(abs(x0[2]),n,vec(w0,w1));
            }
          }
        }
      }
    }
  }
  // Otherwise, minimize over all boundary cases.  This can be further optimized a fair bit, but for now we go for simplicity.
  Tuple<T,TV3,TV2> best;
  {
    const auto r = segment_point_distance_and_normal(s0,s1.x0);
    best = tuple(r.x,TV3(r.y),vec(r.z,0.));
  } {
    const auto r = segment_point_distance_and_normal(s0,s1.x1);
    if (best.x > r.x)
      best = tuple(r.x,TV3(r.y),vec(r.z,1.));
  } {
    const auto r = segment_point_distance_and_normal(s1,s0.x0);
    if (best.x > r.x)
      best = tuple(r.x,TV3(-r.y),vec(0.,r.z));
  } {
    const auto r = segment_point_distance_and_normal(s1,s0.x1);
    if (best.x > r.x)
      best = tuple(r.x,TV3(-r.y),vec(1.,r.z));
  }
  return best;
}

bool segment_ray_intersection(const Segment<TV2>& seg, Ray<TV2>& ray, const T half_thickness) {
  const TV2 from_start_to_start = seg.x0-ray.start;
  TV2 segment_direction = seg.x1-seg.x0;
  const T segment_length = segment_direction.normalize();
  const T cross_product = cross(ray.direction,segment_direction),
          abs_cross_product = abs(cross_product);
  if (ray.t_max*abs_cross_product>half_thickness) {
    const T cross_recip = 1/cross_product;
    const T ray_t = cross_recip*cross(from_start_to_start,segment_direction);
    if (ray_t<0||ray_t>ray.t_max)
      return false;
    const T segment_t = cross_recip*cross(from_start_to_start,ray.direction);
    if (segment_t<-half_thickness || segment_t>segment_length+half_thickness)
      return false;
    ray.t_max = ray_t;
    return true;
  }
  return false;
}

// Generate tetrahedra with random degeneracies
template<int d> static Vector<Vector<T,d>,4> random_degenerate_tetrahedron(Random& random) {
  typedef Vector<T,d> TV;
  Vector<TV,4> X;
  Vector<T,4> w;
  for (const int i : range(4)) {
    X[i] = random.uniform<TV>(-1,1);
    if (random.bit()) {
      // Make X approximately a linear combination of the earlier vertices
      random.fill_uniform(w,log(1e-20),0);
      w = exp(w);
      w /= asarray(w).slice(0,i+1).sum();
      X[i] = w[i]*X[i];
      for (const int j : range(i))
        X[i] += w[j]*X[j];
    }
  }
  random.shuffle(X);
  return X;
}

template<int d> static void segment_tests(const int steps) {
  const T tol = d==2 ? 1e-14 : 1e-13;
  typedef Vector<T,d> TV;
  typedef Vector<T,3> TV3;
  const auto random = new_<Random>(1863418);
  for (int step=0;step<steps;step++) {
    const auto X = random_degenerate_tetrahedron<d>(random);
    const Segment<TV> s0(X[0],X[1]),
                      s1(X[2],X[3]);
    const T len0 = s0.length(),
            len1 = s1.length(),
            min_len = min(len0,len1);
    const Segment<TV3> s0_(TV3(X[0]),(TV3(X[1]))),
                       s1_(TV3(X[2]),(TV3(X[3])));
    const TV p = X[2];
    if (0)
      cout << "\nX = "<<format("[%s,%s,%s,%s]",tuple_repr(X[0]),tuple_repr(X[1]),tuple_repr(X[2]),tuple_repr(X[3]))<<endl;

    // Segment / point
    const T w0 = interpolation_fraction(s0,p);
    const T w1 = clamped_interpolation_fraction(s0,p);
    GEODE_ASSERT(w1==clamp(w0,0.,1.));
    const auto close = segment_closest_point(s0,p);
    const auto c = close.x;
    GEODE_ASSERT(abs(w1-close.y)<tol);
    GEODE_ASSERT(magnitude(c-s0.interpolate(w1))<tol);
    const auto pd = segment_point_distance(s0,p);
    GEODE_ASSERT(abs(pd-magnitude(p-c))<tol);
    const auto r = segment_point_distance_and_normal(s0,p);
    GEODE_ASSERT(abs(pd-r.x)<tol);
    GEODE_ASSERT(abs(magnitude(r.y)-1)<tol);
    GEODE_ASSERT(magnitude(c-p+pd*r.y)<tol);
    GEODE_ASSERT(segment_point_distance(s0,c)<tol);
    GEODE_ASSERT(abs(segment_point_distance(s0,c+3.*r.y)-3)<tol);

    // Line / point
    const T ld = line_point_distance(s0,p);
    GEODE_ASSERT(ld <= pd+tol);
    const auto lclose = line_closest_point(s0,p);
    GEODE_ASSERT(abs(ld-magnitude(p-lclose.x))<tol);
    const auto lr = line_point_distance_and_normal(s0_,TV3(p));
    GEODE_ASSERT(abs(lr.x-ld)*len0<tol);
    GEODE_ASSERT(abs(magnitude(lr.y)-1)<tol);
    GEODE_ASSERT(abs(lr.z-lclose.y)*len0<tol);
    GEODE_ASSERT(magnitude(lclose.x-s0.interpolate(lr.z))<tol);

    // Segment / segment
    const T sd = segment_segment_distance(s0,s1);
    const auto sr = segment_segment_distance_and_normal(s0,s1);
    GEODE_ASSERT(abs(sr.x-sd)<tol);
    const auto n = sr.y;
    GEODE_ASSERT(abs(magnitude(n)-1)<tol);
    const auto w = sr.z;
    GEODE_ASSERT(0<=w.min() && w.max()<=1);
    const auto u = Vector<T,3>(s1.interpolate(w.y)-s0.interpolate(w.x));
    GEODE_ASSERT(abs(sd-magnitude(u))<tol);
    GEODE_ASSERT(magnitude(u-sd*n)<tol);
    const T d0 = dot(n,TV3(s0.vector()));
    // TODO: Make stronger assertions unconditional once the code is fixed
    // See https://github.com/otherlab/geode/issues/42.
    if (sd>.1) {
      if (w.x>0) GEODE_ASSERT(d0>-tol);
      if (w.x<1) GEODE_ASSERT(d0<tol);
    }
    if (0<w.x && w.x<1) GEODE_ASSERT(abs(d0)<tol);
    const T d1 = -dot(n,TV3(s1.vector()));
    // TODO: Make stronger assertions unconditional once the code is fixed
    // See https://github.com/otherlab/geode/issues/42.
    if (sd>.1) {
      if (w.y>0) GEODE_ASSERT(d1>-tol);
      if (w.y<1) GEODE_ASSERT(d1<tol);
    }
    if (0<w.y && w.y<1) GEODE_ASSERT(abs(d1)<tol);

    // Line / line
    if (d==3) {
      const auto r = line_line_distance_and_normal(s0_,s1_);
      GEODE_ASSERT(r.x <= sd+tol);
      const auto n = r.y;
      const auto w = r.z;
      const auto u = s1_.interpolate(w.y)-s0_.interpolate(w.x);
      GEODE_ASSERT(abs(r.x-magnitude(u))*min_len<tol);
      GEODE_ASSERT(magnitude(u-r.x*n)*min_len<tol);
      GEODE_ASSERT(abs(dot(n,s0_.vector()))<tol);
      GEODE_ASSERT(abs(dot(n,s1_.vector()))<tol);
    }
  }
}

#define INSTANTIATE(d) \
  template GEODE_CORE_EXPORT Tuple<Vector<T,d>,T> segment_closest_point(Segment<Vector<T,d>>,Vector<T,d>); \
  template GEODE_CORE_EXPORT T interpolation_fraction(const Segment<Vector<T,d>>&,const Vector<T,d>); \
  template GEODE_CORE_EXPORT T clamped_interpolation_fraction(const Segment<Vector<T,d>>&,const Vector<T,d>); \
  template GEODE_CORE_EXPORT T segment_point_distance(Segment<Vector<T,d>>,Vector<T,d>); \
  template GEODE_CORE_EXPORT T segment_point_sqr_distance(Segment<Vector<T,d>>,Vector<T,d>); \
  template GEODE_CORE_EXPORT T segment_segment_distance(Segment<Vector<T,d>>,Segment<Vector<T,d>>); \
  template GEODE_CORE_EXPORT Tuple<T,Vector<T,d>,T> segment_point_distance_and_normal(Segment<Vector<T,d>> s, Vector<T,d> p); \
  template GEODE_CORE_EXPORT Tuple<T,TV3,TV2> segment_segment_distance_and_normal(const Segment<Vector<T,d>>,const Segment<Vector<T,d>>); \
  template GEODE_CORE_EXPORT T line_point_distance(Segment<Vector<T,d>>,Vector<T,d>);
INSTANTIATE(2)
INSTANTIATE(3)

}
using namespace geode;

void wrap_segment() {
  GEODE_FUNCTION_2(segment_tests_2d,segment_tests<2>)
  GEODE_FUNCTION_2(segment_tests_3d,segment_tests<3>)
}
