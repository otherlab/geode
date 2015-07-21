#include <geode/solver/pattern_max.h>
#include <geode/python/function.h>
#include <geode/python/wrap.h>


namespace geode {

real spherical_pattern_maximize(const function<real(Vector<real,3>)>& score, Vector<real,3>& n, real tol) {
  static const real da = 2*M_PI/5;
  static const Vector<real,2> dirs[5] = {polar(0.), polar(da), polar(2*da), polar(3*da), polar(4*da)};
  const real alpha = .5;
  real step = .2;
  real dot = score(n);
  while (step > tol) {
    real best_dot = dot;
    Vector<real,3> best_n = n;
    Vector<real,3> orth;
    orth[n.argmin()] = 1;
    Vector<real,3> a = cross(n,orth).normalized(), b = cross(n,a);
    for (int i = 0; i < 5; i++) {
      Vector<real,3> candidate = (n + step*dirs[i].x*a + step*dirs[i].y*b).normalized();
      real d = score(candidate);
      if (best_dot < d) {
        best_dot = d;
        best_n = candidate;
      }
    }
    if (dot < best_dot) {
      dot = best_dot;
      n = best_n;
    } else
      step *= alpha;
  }
  return dot;
}

}

using namespace geode;

void wrap_pattern_max() {
  GEODE_FUNCTION(spherical_pattern_maximize)
}
