#include <geode/solver/quadratic.h>
#include <geode/math/robust.h>
namespace geode {

SmallArray<real,3> solve_cubic(const real a, const real b, const real c, const real d) {
  SmallArray<real,3> result;

  const auto a2 = sqr(a);
  const auto a3 = a*a2;

  const auto q = (a2 - 3*b) / 9;
  const auto r = (2*a3 - 9*a*b + 27*c) / 54;
  const auto r2 = sqr(r);
  const auto q3 = cube(r);

  if(r2 < q3) {
    // Note: For real inputs, sqr(r) is always non-negative and since r2 < q3, q3 >= 0 and thus q >= 0
    const auto theta = acos(r/sqrt(q3));
    const auto sqrt_q = sqrt(q);
    // Three roots, all real
    result.append(-2*sqrt_q*cos(theta/3) - a/3);
    result.append(-2*sqrt_q*cos((theta + tau)/3) - a/3);
    result.append(-2*sqrt_q*cos((theta - tau)/3) - a/3);
    small_sort(result[0],result[1], result[2]);
  }
  else {
    const auto A = -copysign(cbrt(abs(r) + sqrt(r2-q3)), r);
    const auto B = pseudo_divide(q, A);
    // One real root...
    result.append(A + B - a/3);
    // ...and two complex roots which we ignore
    // -0.5*(A+B) - a/3. + i*(sqrt(3.)/2.)*(A-B)
    // -0.5*(A+B) - a/3. - i*(sqrt(3.)/2.)*(A-B)
  }

  return result;
}

} // geode namespace