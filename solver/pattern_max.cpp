#include <other/core/vector/Vector.h>
namespace other{
// Maximize a functional over a sphere using pattern search
template<class score_t> static double spherical_pattern_maximize(const score_t& score, Vector<real,3>& n, double tol) {
  static const double da = 2*M_PI/5;
  static const Vector<real,2> dirs[5] = {polar(0.), polar(da), polar(2*da), polar(3*da), polar(4*da)};
  const double alpha = .5;
  double step = .2;
  double dot = score(n);
  while (step > tol) {
    double best_dot = dot;
    Vector<real,3> best_n = n;
    Vector<real,3> orth;
    orth[n.argmin()] = 1;
    Vector<real,3> a = cross(n,orth).normalized(), b = cross(n,a);
    for (int i = 0; i < 5; i++) {
      Vector<real,3> candidate = (n + step*dirs[i].x*a + step*dirs[i].y*b).normalized();
      double d = score(candidate);
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
