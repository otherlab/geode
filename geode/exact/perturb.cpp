// General purpose black box simulation of simplicity

#include <geode/exact/perturb.h>
#include <geode/exact/math.h>
#include <geode/exact/polynomial.h>
#include <geode/array/alloca.h>
#include <geode/array/Array2d.h>
#include <geode/array/Array3d.h>
#include <geode/python/wrap.h>
#include <geode/random/counter.h>
#include <geode/utility/move.h>
#include <geode/vector/Matrix.h>
namespace geode {

// Our function is defined by
//
//   perturbed_sign(f,_,x) = lim_{ek -> 0} (f(x + sum_{k>0} ek yk) > 0)
//
// where yk are fixed pseudorandom vectors and ei >> ej for i > j in the limit.  Almost all of the time,
// the first e1 y1 term is sufficient to reach nondegeneracy, so the practical complexity is O(predicate-cost*degree).
// Our scheme is a combination of the fully general scheme of Yap and the randomized linear scheme of Seidel:
//
//   Yap 1990, "Symbolic treatment of geometric degeneracies".
//   Seidel 1998, "The nature and meaning of perturbations in geometric computing".
//
// To recover the expanded predicate at each level of the interpolation, we use the divided difference algorithm of
//
//   Neidinger 2010, "Multivariable interpolating polynomials in Newton forms".
//
// In their terminology, we evaluate polynomials on "easy corners" where x_i(j) = j.  In the univariate case we
// precompute the LU decomposition of the Vandermonde matrix, invert each part, and clear fractions to avoid the need
// for rational arithmetic.  The structure of the LU decomposition of the Vandermonde matrix is given in
//
//   Oliver 2009, "On multivariate interpolation".

using std::cout;
using std::endl;

// Compile time debugging support
static const bool check = false;
static const bool verbose = false;

// Our fixed deterministic pseudorandom perturbation sequence.  We limit ourselves to 32 bits so that we can pull four values out of a uint128_t.
template<int m> inline Vector<ExactInt,m> perturbation(const int level, const int i) {
  static_assert(m<=4,"");
  const int bits = min(exact::log_bound+1,128/4);
  const auto limit = ExactInt(1)<<(bits-1);
  const uint128_t noise = threefry(level,i);
  Vector<ExactInt,m> result;
  for (int a=0;a<m;a++)
    result[a] = (cast_uint128<ExactInt>(noise>>32*a)&(2*limit-1))-limit;
  return result;
}

/********** Symbolic perturbation **********/

template<int m> static inline Vector<ExactInt,m> to_exact(const Vector<Quantized,m>& x) {
  Vector<ExactInt,m> r;
  for (int i=0;i<m;i++) {
    r[i] = ExactInt(x[i]);
    assert(r[i]==x[i]); // Make sure the input is actually quantized
  }
  return r;
}

static bool last_nonzero(RawArray<const mp_limb_t> x) {
  return mpz_nonzero(x);
}

static bool last_nonzero(RawArray<const mp_limb_t,2> x) {
  return mpz_nonzero(x[x.m-1]);
}

// Check for identically zero polynomials using randomized polynomial identity testing
template<class R,int m> static void
assert_last_nonzero(void(*const polynomial)(R,RawArray<const Vector<Exact<1>,m>>),
                    R result, RawArray<const Tuple<int,Vector<Quantized,m>>> X, const char* message) {
  typedef Vector<Exact<1>,m> EV;
  const int n = X.size();
  const auto Z = GEODE_RAW_ALLOCA(n,EV);
  for (const int k : range(20)) {
    for (int i=0;i<n;i++)
      Z[i] = EV(perturbation<m>(k<<10,X[i].x));
    polynomial(result,Z);
    if (last_nonzero(result)) // Even a single nonzero means we're all good
      return;
  }
  // If we reach this point, the chance of a nonzero nonmalicious polynomial is something like (1e-5)^20 = 1e-100.  Thus, we can safely assume that for the lifetime
  // of this code, we will never treat a nonzero polynomial as zero.  If this comes up, we can easily bump the threshold a bit further.
  throw AssertionError(format("%s (there is likely a bug in the calling code), X = %s",message,str(X)));
}

template<int m> bool perturbed_sign(void(*const predicate)(RawArray<mp_limb_t>,RawArray<const Vector<Exact<1>,m>>),
                                    const int degree, RawArray<const Tuple<int,Vector<Quantized,m>>> X) {
  typedef Vector<Exact<1>,m> EV;
  if (check)
    GEODE_WARNING("Expensive consistency checking enabled");

  const int n = X.size();
  if (verbose)
    cout << "perturbed_sign:\n  degree = "<<degree<<"\n  X = "<<X<<endl;

  // Check if the predicate is nonsingular without perturbation
  const auto Z = GEODE_RAW_ALLOCA(n,EV);
  const int precision = degree*Exact<1>::ratio;
  {
    for (int i=0;i<n;i++)
      Z[i] = EV(to_exact(X[i].y));
    const auto R = GEODE_RAW_ALLOCA(precision,mp_limb_t);
    predicate(R,Z);
    if (const int sign = mpz_sign(R))
      return sign>0;
  }

  // Check the first perturbation level with specialized code
  vector<Vector<ExactInt,m>> Y(n); // perturbations
  {
    // Compute the first level of perturbations
    for (int i=0;i<n;i++)
      Y[i] = perturbation<m>(1,X[i].x);
    if (verbose)
      cout << "  Y = "<<Y<<endl;

    // Evaluate polynomial at epsilon = 1, ..., degree
    const int scaled_precision = precision+factorial_limbs(degree);
    const auto values = GEODE_RAW_ALLOCA(degree*scaled_precision,mp_limb_t).reshape(degree,scaled_precision);
    memset(values.data(),0,sizeof(mp_limb_t)*values.flat.size());
    for (int j=0;j<degree;j++) {
      for (int i=0;i<n;i++)
        Z[i] = EV(to_exact(X[i].y)+(j+1)*Y[i]);
      predicate(values[j],Z);
      if (verbose)
        cout << "  predicate("<<Z<<") = "<<mpz_str(values[j])<<endl;
    }

    // Find an interpolating polynomial, overriding the input with the result.
    scaled_univariate_in_place_interpolating_polynomial(values);
    if (verbose)
      cout << "  coefs = "<<mpz_str(values)<<endl;

    // Compute sign
    for (int j=0;j<degree;j++)
      if (const int sign = mpz_sign(values[j]))
        return sign>0;
  }

  {
    // Add one perturbation level after another until we hit a nonzero polynomial.  Our current implementation duplicates
    // work from one iteration to the next for simplicity, which is fine since the first interation suffices almost always.
    for (int d=2;;d++) {
      if (verbose)
        cout << "  level "<<d<<endl;
      // Compute the next level of perturbations
      Y.resize(d*n);
      for (int i=0;i<n;i++)
        Y[(d-1)*n+i] = perturbation<m>(d,X[i].x);

      // Evaluate polynomial at every point in an "easy corner"
      const auto lambda = monomials(degree,d);
      const Array<mp_limb_t,2> values(lambda.m,precision,uninit);
      for (int j=0;j<lambda.m;j++) {
        for (int i=0;i<n;i++)
          Z[i] = EV(to_exact(X[i].y)+lambda(j,0)*Y[i]);
        for (int v=1;v<d;v++)
          for (int i=0;i<n;i++)
            Z[i] += EV(lambda(j,v)*Y[v*n+i]);
        predicate(values[j],Z);
      }

      // Find an interpolating polynomial, overriding the input with the result.
      in_place_interpolating_polynomial(degree,lambda,values);

      // Compute sign
      int sign = 0;
      int sign_j = -1;
      for (int j=0;j<lambda.m;j++)
        if (const int s = mpz_sign(values[j])) {
          if (check) // Verify that a term which used to be zero doesn't become nonzero
            GEODE_ASSERT(lambda(j,d-1));
          if (!sign || monomial_less(lambda[sign_j],lambda[j])) {
            sign = s;
            sign_j = j;
          }
        }

      // If we find a nonzero sign, we're done!
      if (sign)
        return sign>0;

      // If we get through two levels without fixing the degeneracy, run a fast, strict identity test to make sure we weren't handed an impossible problem.
      if (d==2)
        assert_last_nonzero(predicate,values[0],X,"perturbed_sign: identically zero predicate");
    }
  }
}

static inline RawArray<mp_limb_t> sqrt_helper(RawArray<mp_limb_t> result, RawArray<const mp_limb_t> x) {
  const auto s = result.slice(0,(1+x.size())/2);
  mpn_sqrtrem(s.data(),0,x.data(),x.size());
  return trim(s);
}

// Cast num/den to an int, rounding towards nearest.  All inputs are destroyed.  Take a sqrt if desired.
// The values array must consist of r numerators followed by one denominator.
void snap_divs(RawArray<Quantized> result, RawArray<mp_limb_t,2> values, const bool take_sqrt) {
  assert(result.size()+1==values.m);

  // For division, we seek x s.t.
  //   x-1/2 <= num/den <= x+1/2
  //   2x-1 <= 2num/den <= 2x+1
  //   2x-1 <= floor(2num/den) <= 2x+1
  //   2x <= 1+floor(2num/den) <= 2x+2
  //   x <= (1+floor(2num/den))//2 <= x+1
  //   x = (1+floor(2num/den))//2

  // In the sqrt case, we seek a nonnegative integer x s.t.
  //   x-1/2 <= sqrt(num/den) < x+1/2
  //   2x-1 <= sqrt(4num/den) < 2x+1
  // Now the leftmost and rightmost expressions are integral, so we can take floors to get
  //   2x-1 <= floor(sqrt(4num/den)) < 2x+1
  // Since sqrt is monotonic and maps integers to integers, floor(sqrt(floor(x))) = floor(sqrt(x)), so
  //   2x-1 <= floor(sqrt(floor(4num/den))) < 2x+1
  //   2x <= 1+floor(sqrt(floor(4num/den))) < 2x+2
  //   x <= (1+floor(sqrt(floor(4num/den))))//2 < x+1
  //   x = (1+floor(sqrt(floor(4num/den))))//2

  // Thus, both cases look like
  //   x = (1+f(2**k*num/den))//2
  // where k = 1 or 2 and f is some truncating integer op (division or division+sqrt).

  // Adjust denominator to be positive
  const auto raw_den = values[result.size()];
  const bool den_negative = mp_limb_signed_t(raw_den.back())<0;
  if (den_negative)
    mpn_neg(raw_den.data(),raw_den.data(),raw_den.size());
  const auto den = trim(raw_den);
  assert(den.size()); // Zero should be prevented by the caller

  // Prepare for divisions
  const auto q = GEODE_RAW_ALLOCA(values.n-den.size()+1,mp_limb_t),
             r = GEODE_RAW_ALLOCA(den.size(),mp_limb_t);

  // Compute each component of the result
  for (int i=0;i<result.size();i++) {
    // Adjust numerator to be positive
    const auto num = values[i];
    const bool num_negative = mp_limb_signed_t(num.back())<0;
    if (take_sqrt && num_negative!=den_negative && !num.contains_only(0))
      throw RuntimeError("perturbed_ratio: negative value in square root");
    if (num_negative)
      mpn_neg(num.data(),num.data(),num.size());

    // Add enough bits to allow round-to-nearest computation after performing truncating operations
    mpn_lshift(num.data(),num.data(),num.size(),take_sqrt?2:1);
    // Perform division
    mpn_tdiv_qr(q.data(),r.data(),0,num.data(),num.size(),den.data(),den.size());
    const auto trim_q = trim(q);
    if (!trim_q.size()) {
      result[i] = 0;
      continue;
    }
    // Take sqrt if desired, reusing the num buffer
    const auto s = take_sqrt ? sqrt_helper(num,trim_q) : trim_q;

    // Verify that result lies in [-exact::bound,exact::bound];
    const int ratio = sizeof(ExactInt)/sizeof(mp_limb_t);
    static_assert(ratio<=2,"");
    if (s.size() > ratio)
      goto overflow;
    const auto nn = ratio==2 && s.size()==2 ? s[0]|ExactInt(s[1])<<8*sizeof(mp_limb_t) : s[0],
               n = (1+nn)/2;
    if (uint64_t(n) > uint64_t(exact::bound))
      goto overflow;

    // Done!
    result[i] = (num_negative==den_negative?1:-1)*Quantized(n);
  }

  return;
  overflow:
  throw OverflowError("perturbed_ratio: overflow in l'Hopital expansion");
}

template<int m> bool perturbed_ratio(RawArray<Quantized> result, void(*const ratio)(RawArray<mp_limb_t,2>,RawArray<const Vector<Exact<1>,m>>), const int degree, RawArray<const Tuple<int,Vector<Quantized,m>>> X, const bool take_sqrt) {
  typedef Vector<Exact<1>,m> EV;
  const int n = X.size();
  const int r = result.size();

  if (verbose)
    cout << "perturbed_ratio:\n  degree = "<<degree<<"\n  X = "<<X<<endl;

  // Check if the ratio is nonsingular before perturbation
  const auto Z = GEODE_RAW_ALLOCA(n,EV);
  const int precision = degree*Exact<1>::ratio;
  {
    for (int i=0;i<n;i++)
      Z[i] = EV(to_exact(X[i].y));
    const auto R = GEODE_RAW_ALLOCA((r+1)*precision,mp_limb_t).reshape(r+1,precision);
    ratio(R,Z);
    if (const int sign = mpz_sign(R[r])) {
      snap_divs(result,R,take_sqrt);
      return sign>0;
    }
  }

  // Check the first perturbation level with specialized code
  vector<Vector<ExactInt,m>> Y(n); // perturbations
  {
    // Compute the first level of perturbations
    for (int i=0;i<n;i++)
      Y[i] = perturbation<m>(1,X[i].x);
    if (verbose)
      cout << "  Y = "<<Y<<endl;

    // Evaluate polynomial at epsilon = 1, ..., degree
    const int scaled_precision = precision+factorial_limbs(degree);
    const auto values = GEODE_RAW_ALLOCA(degree*(r+1)*scaled_precision,mp_limb_t).reshape(degree,r+1,scaled_precision);
    for (int j=0;j<degree;j++) {
      for (int i=0;i<n;i++)
        Z[i] = EV(to_exact(X[i].y)+(j+1)*Y[i]);
      ratio(values[j],Z);
      if (verbose)
        cout << "  ratio("<<Z<<") = "<<mpz_str(values[j])<<endl;
    }

    // Find interpolating polynomials, overriding the input with the result.
    for (int k=0;k<=r;k++) {
      scaled_univariate_in_place_interpolating_polynomial(values.sub<1>(k));
      if (verbose)
        cout << "  coefs "<<k<<" = "<<mpz_str(values.sub<1>(k))<<endl;
    }

    // Find the largest (lowest degree) nonzero denominator coefficient.  If we detect an infinity during this process, explode.
    for (int j=0;j<degree;j++) {
      if (const int sign = mpz_sign(values(j,r))) { // We found a nonzero, now compute the rounded ratio
        snap_divs(result,values[j],take_sqrt);
        return sign>0;
      } else
        for (int k=0;k<r;k++)
          if (mpz_nonzero(values(j,k)))
            throw OverflowError(format("perturbed_ratio: infinite result in l'Hopital expansion: %s/0",mpz_str(values(j,k))));
    }
  }

  {
    // Add one perturbation level after another until we hit a nonzero denominator.  Our current implementation duplicates
    // work from one iteration to the next for simplicity, which is fine since the first interation suffices almost always.
    for (int d=2;;d++) {
      // Compute the next level of perturbations
      Y.resize(d*n);
      for (int i=0;i<n;i++)
        Y[(d-1)*n+i] = perturbation<m>(d,X[i].x);

      // Evaluate polynomial at every point in an "easy corner"
      const auto lambda = monomials(degree,d);
      const Array<mp_limb_t,3> values(lambda.m,r+1,precision,uninit);
      for (int j=0;j<lambda.m;j++) {
        for (int i=0;i<n;i++)
          Z[i] = EV(to_exact(X[i].y)+lambda(j,0)*Y[i]);
        for (int v=1;v<d;v++)
          for (int i=0;i<n;i++)
            Z[i] += EV(lambda(j,v)*Y[v*n+i]);
        ratio(values[j],Z);
      }

      // Find interpolating polynomials, overriding the input with the result.
      for (int k=0;k<=r;k++)
        in_place_interpolating_polynomial(degree,lambda,values.sub<1>(k));

      // Find the largest nonzero denominator coefficient
      int sign = 0;
      int nonzero = -1;
      for (int j=0;j<lambda.m;j++)
        if (const int s = mpz_sign(values(j,r))) {
          if (check) // Verify that a term which used to be zero doesn't become nonzero
            GEODE_ASSERT(lambda(j,d-1));
          if (nonzero<0 || monomial_less(lambda[nonzero],lambda[j])) {
            sign = s;
            nonzero = j;
          }
        }

      // Verify that numerator coefficients are zero for all large monomials
      for (int j=0;j<lambda.m;j++)
        if (nonzero<0 || monomial_less(lambda[nonzero],lambda[j]))
          for (int k=0;k<r;k++)
            if (mpz_nonzero(values(j,k)))
              throw OverflowError(format("perturbed_ratio: infinite result in l'Hopital expansion: %s/0",str(values(j,k))));

      // If we found a nonzero, compute the result
      if (nonzero >= 0) {
        snap_divs(result,values[nonzero],take_sqrt);
        return sign>0;
      }

      // If we get through two levels without fixing the degeneracy, run a fast, strict identity test to make sure we weren't handed an impossible problem.
      if (d==2)
        assert_last_nonzero(ratio,values[0],X,"perturbed_ratio: identically zero denominator");
    }
  }
}

// Everything that follows is for testing purposes

// Safely expose snap_divs to python for testing purposes
static Array<Quantized> snap_divs_test(RawArray<mp_limb_t,2> values, const bool take_sqrt) {
  GEODE_ASSERT(values.m && !values.back().contains_only(0));
  Array<Quantized> result(values.m-1);
  snap_divs(result,values,take_sqrt);
  return result;
}

// Test against malicious predicates that are zero along 0, 1, or 2 perturbation levels.

static int nasty_index, nasty_power;

template<int d> static void nasty_pow(RawArray<mp_limb_t> result, const Exact<d>& x) {
  switch (nasty_power) {
    case 1: mpz_set(result,x);       break;
    case 2: mpz_set(result,sqr(x));  break;
    case 3: mpz_set(result,cube(x)); break;
    default: GEODE_FATAL_ERROR();
  };
}

template<class In> static void nasty_predicate(RawArray<mp_limb_t> result, RawArray<const Vector<In,1>> X) {
  nasty_pow(result,X[0].x);
}

template<class In> static void nasty_predicate(RawArray<mp_limb_t> result, RawArray<const Vector<In,2>> X) {
  typename remove_const_reference<decltype(X[0])>::type p1;
  for (int i=0;i<2;i++)
    mpz_set(asarray(p1[i].n),Exact<1>(perturbation<2>(1,nasty_index)[i]));
  nasty_pow(result,edet(X[0],p1));
}

template<class In> static void nasty_predicate(RawArray<mp_limb_t> result, RawArray<const Vector<In,3>> X) {
  typename remove_const_reference<decltype(X[0])>::type p1,p2;
  for (int i=0;i<3;i++) {
    mpz_set(asarray(p1[i].n),Exact<1>(perturbation<3>(1,nasty_index)[i]));
    mpz_set(asarray(p2[i].n),Exact<1>(perturbation<3>(2,nasty_index)[i]));
  }
  nasty_pow(result,edet(X[0],p1,p2));
}

template<int m> static void perturbed_sign_test() {
  for (const int power : vec(1,2,3))
    for (const int index : range(20)) {
      if (verbose)
        cout << endl;
      // Evaluate perturbed sign using our fancy routine
      nasty_power = power;
      nasty_index = index;
      Array<Tuple<int,Vector<Quantized,m>>> fX(1);
      fX[0].x = index;
      const bool fast = perturbed_sign<m>(nasty_predicate,m*power,fX);
      GEODE_ASSERT((power&1) || fast);
      // Evaluate the series out to several terms using brute force
      Array<int> powers(m+1); // Choose powers of 2 to approximate nested infinitesimals
      for (int i=0;i<m;i++)
        powers[i+1] = (power+1)*powers[i]+128;
      mpz_t yp;
      mpz_init(yp);
      const int enough = 5120/(8*sizeof(ExactInt));
      Vector<Exact<enough>,m> sX[1];
      for (int i=0;i<=m+1;i++) {
        if (i) {
          const Vector<Exact<1>,m> y(perturbation<m>(i,index));
          for (int j=0;j<m;j++) {
            auto& x = sX[0][j];
            Exact<enough> yp;
            const int skip = (powers.back()-powers[i-1])/(8*sizeof(mp_limb_t));
            mpz_set(asarray(yp.n).slice(skip,yp.limbs),y[j]); // yp = y[j]<<(powers[-1]-powers[i-1])
            x += yp;
          }
        }
        // We should be initially zero, and then match the correct sign once nonzero
        Array<mp_limb_t> result(enough*m*power,uninit);
        nasty_predicate(result,asconstarray(sX));
        const int slow = mpz_sign(result);
        if (0) {
          cout << "m "<<m<<", power "<<power<<", index "<<index<<", i "<<i<<", fast "<<2*fast-1<<", slow "<<slow<<endl;
          cout << "  fX = "<<fX[0]<<", sX = "<<sX[0]<<" (x = "<<mpz_str(asarray(sX[0].x.n),true)<<')'<<endl;
          cout << "  sX result = "<<mpz_str(result,true)<<endl;
        }
        GEODE_ASSERT(slow==(i<m?0:2*fast-1));
      }
      mpz_clear(yp);
    }
}

// The unit tests in constructions.cpp and circle_csg.cpp are fairly rigorous checks of the geometric validity
// perturbed ratio to levels 0 and 1, but are unlikely to ever hit perturbation level 2 or higher.  Therefore,
// we construct and test a malicious predicate guaranteed to hit level 2.

static void nasty_ratio(RawArray<mp_limb_t,2> result, RawArray<const Vector<Exact<1>,2>> X) {
  assert(result.m==2 && X.size()==2);
  typename remove_const_reference<decltype(X[0])>::type p1;
  for (int i=0;i<2;i++)
    mpz_set(asarray(p1[i].n),Exact<1>(perturbation<2>(1,nasty_index)[i]));
  const auto d = edet(X[0],p1);
  if (nasty_power==1)
    mpz_set(result[0],d*X[1].x);
  else
    mpz_set(result[0],d*sqr(X[1].x));
  mpz_set(result[1],d);
}
static void nasty_denominator(RawArray<mp_limb_t> result, RawArray<const Vector<Exact<1>,2>> X) {
  assert(X.size()==2);
  typename remove_const_reference<decltype(X[0])>::type p1;
  for (int i=0;i<2;i++)
    mpz_set(asarray(p1[i].n),Exact<1>(perturbation<2>(1,nasty_index)[i]));
  const auto d = edet(X[0],p1);
  mpz_set(result,d);
}

static void perturbed_ratio_test() {
  typedef Vector<Quantized,2> EV;
  for (const int power : vec(1,2))
    for (const int index : range(20)) {
      nasty_power = power;
      nasty_index = index;
      Vector<const Tuple<int,EV>,2> X(tuple(index,EV()),tuple(index+1,EV(perturbation<2>(7,index+1))));
      Vector<Quantized,1> result;
      const bool s = perturbed_ratio(asarray(result),nasty_ratio,2+power,asarray(X),power==2);
      GEODE_ASSERT(result.x==(power==1?X.y.y.x:abs(X.y.y.x)));
      GEODE_ASSERT(s==perturbed_sign(nasty_denominator,1+power,asarray(X)));
    }
}

#define INSTANTIATE(m) \
  template Vector<ExactInt,m> perturbation(const int, const int); \
  template bool perturbed_sign(void(*const)(RawArray<mp_limb_t>,RawArray<const Vector<Exact<1>,m>>), \
                                            const int, RawArray<const Tuple<int,Vector<Quantized,m>>>); \
  template bool perturbed_ratio(RawArray<Quantized>,void(*const)(RawArray<mp_limb_t,2>, \
                                RawArray<const Vector<Exact<1>,m>>), const int, \
                                RawArray<const Tuple<int,Vector<Quantized,m>>>, bool);
INSTANTIATE(1)
INSTANTIATE(2)
INSTANTIATE(3)

}
using namespace geode;

void wrap_perturb() {
  GEODE_FUNCTION_2(perturbed_sign_test_1,perturbed_sign_test<1>)
  GEODE_FUNCTION_2(perturbed_sign_test_2,perturbed_sign_test<2>)
  GEODE_FUNCTION_2(perturbed_sign_test_3,perturbed_sign_test<3>)
  GEODE_FUNCTION(snap_divs_test)
  GEODE_FUNCTION(perturbed_ratio_test)
}
