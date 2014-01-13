// General purpose black box simulation of simplicity

#include <geode/exact/perturb.h>
#include <geode/exact/math.h>
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

// List all n-variate monomials of degree <= d, ordered by ascending total degree and then arbitrarily.
// Warning: This is the order needed for divided difference interpolation, but is *not* the correct infinitesimal size order.
static Array<const uint8_t,2> monomials(const int degree, const int variables) {
  // Count monomials: choose(degree+variables,degree)
  uint64_t num = 1, den = 1;
  for (int k=1;k<=degree;k++) {
    num *= k+variables;
    den *= k;
  }
  num /= den;
  GEODE_ASSERT(num <= (1<<20));
  Array<uint8_t,2> results((int)num,variables);

  // We simulate a stack manually to avoid recursion
  if (variables) {
    int next = 0;
    const auto alpha = GEODE_RAW_ALLOCA(variables,uint8_t);
    alpha.fill(0);
    for (int d=0;d<=degree;d++) {
      int i = 0;
      int left = d;
      alpha[0] = 0;
      for (;;) {
        if (i<variables-1) {
          // Traverse down
          i++;
        } else {
          // Add our complete monomial to the list
          alpha[i] = left;
          results[next++] = alpha;
          // Traverse up until we can increment alpha[i]
          for (;;) {
            if (!i--)
              goto end;
            if (!left) {
              left += alpha[i];
              alpha[i] = 0;
            } else {
              left--;
              alpha[i]++;
              break;
            }
          }
        }
      }
      end:;
    }
    assert(next==results.m);
  }
  return results;
}

static string show_monomial(RawArray<const uint8_t> alpha) {
  string s(alpha.size(),'0');
  for (int i=0;i<alpha.size();i++)
    s[i] = '0'+alpha[i];
  return s;
}

// The relative size ordering on infinitesimals
static inline bool monomial_less(RawArray<const uint8_t> a, RawArray<const uint8_t> b) {
  assert(a.size()==b.size());
  for (int i=a.size()-1;i>=0;i--)
    if (a[i] != b[i])
      return a[i]>b[i];
  return false;
}

/********** Integer arithmetic with manual memory management for speed **********/

static int mpz_sign(RawArray<const mp_limb_t> x) {
  if (mp_limb_signed_t(x.back())<0)
    return -1;
  for (int i=0;i<x.size();i++)
    if (x[i])
      return 1;
  return 0;
}

static bool mpz_nonzero(RawArray<const mp_limb_t> x) {
  return !x.contains_only(0);
}

// x <<= count
static inline void mpz_lshift(RawArray<mp_limb_t> x, const int count) {
  assert(0<count && count<int(8*sizeof(mp_limb_t)));
  mpn_lshift(x.data(),x.data(),x.size(),count);
}

// x += y
static inline void mpz_add(RawArray<mp_limb_t> x, RawArray<const mp_limb_t> y) {
  assert(x.size()==y.size());
  mpn_add_n(x.data(),x.data(),y.data(),x.size());
}

// x -= y
static inline void mpz_sub(RawArray<mp_limb_t> x, RawArray<const mp_limb_t> y) {
  assert(x.size()==y.size());
  mpn_sub_n(x.data(),x.data(),y.data(),x.size());
}

// x *= s
static inline void mpz_mul_ui(RawArray<mp_limb_t> x, const mp_limb_t s) {
  mpn_mul_1(x.data(),x.data(),x.size(),s);
}

// x -= s*y
static inline void mpz_submul_ui(RawArray<mp_limb_t> x, RawArray<const mp_limb_t> y, const mp_limb_t s) {
  assert(x.size()==y.size());
  mpn_submul_1(x.data(),y.data(),x.size(),s);
}

// x /= s, asserting exact divisibility
static inline void mpz_div_exact_ui(RawArray<mp_limb_t> x, const mp_limb_t s) {
  const bool negative = mp_limb_signed_t(x.back())<0;
  if (negative)
    mpn_neg(x.data(),x.data(),x.size());
  GEODE_DEBUG_ONLY(const auto rem =) mpn_divmod_1(x.data(),x.data(),x.size(),s);
  assert(!rem);
  if (negative)
    mpn_neg(x.data(),x.data(),x.size());
}

/********** Polynomial interpolation **********/

// Given the values of a polynomial at every point in the standard "easy corner", solve for the monomial coefficients using divided differences
// lambda and A are as in Neidinger.  We assume lambda is partially sorted by total degree.
static void in_place_interpolating_polynomial(const int degree, const RawArray<const uint8_t,2> lambda, Subarray<mp_limb_t,2> A) {
  // For now we are lazy, and index using a rectangular helper array mapping multi-indices to flat indices
  const int n = lambda.n;
  Array<int> powers(n+1,false);
  powers[0] = 1;
  for (int i=0;i<n;i++)
    powers[i+1] = powers[i]*(degree+1);
  Array<uint16_t> to_flat(powers.back(),false);
  Array<int> from_flat(lambda.m,false);
  to_flat.fill(-1);
  for (int k=0;k<lambda.m;k++) {
    int f = 0;
    for (int i=0;i<n;i++)
      f += powers[i]*lambda(k,i);
    from_flat[k] = f;
    to_flat[f] = k;
  }

  // Bookkeeping information for the divided difference algorithm
  Array<Vector<int,2>> info(lambda.m,false); // m,alpha[m] for each tick
  for (int k=0;k<lambda.m;k++)
    info[k].set(0,lambda(k,0));
  // In self check mode, keep track of the entire alpha
  Array<uint8_t,2> alpha;
  if (check)
    alpha = lambda.copy();

  // Iterate divided differences for degree = max |lambda| passes.
  for (int pass=1;pass<=degree;pass++) {
    for (int k=lambda.m-1;k>=0;k--) {
      // Decrement alpha
      auto& I = info[k];
      while (!I.y) {
        if (++I.x==n) // Quit if we're done with this degree, since lambda[k] for smaller k will also be finished.
          goto next_pass;
        I.y = lambda(k,I.x);
      }
      I.y--;
      // Compute divided difference
      const int child = to_flat[from_flat[k]-powers[I.x]];
      mpz_sub(A[k],A[child]); // A[k] -= A[child]
      mpz_div_exact_ui(A[k],lambda(k,I.x)-I.y); // A[k] /= lambda(k,I.x)-I.y
      // In self check mode, verify that the necessary f[alpha,beta] values were available
      if (check) {
        alpha(k,I.x)--;
        GEODE_ASSERT(alpha[k]==alpha[child]);
      }
    }
    next_pass:;
  }

  // At this point A contains the coefficients of the interpolating polynomial in the Newton basis, and we are halfway there.  Next, we must
  // expand the Newton basis out into the monomial basis.  We start with A(beta) = g(beta), the coefficient of the Newton basis polynomial q_beta.
  // We seek h(alpha) for alpha <= beta, the coefficients of the monomial basis polynomials x^alpha.  This gives rise to a special upper triangular
  // matrix M = M_{alpha,beta}:
  //
  //   h = M g
  //   M_ab = 0 unless a <= b componentwise: M is upper triangular w.r.t. the partial order on ticks
  //   M_bb = 1: M is special upper triangular, since Newton basis polynomials are monic
  //
  // Since the multivariate Newton polynomials are simple products of univariate Newton polynomials for each variable, M factors into n commuting
  // block diagonal matrices M_i for each variable x_i, corresponding to expanding one variable while leaving the others unchanged.  Each M_i further
  // factors into a product of special upper bidiagonal matrices U_{i,j}^{-1}, as given in the univariate case below.
  //
  // Note: Since lambda is not ordered with respect to any single variable, it is difficult to take advantage of the sparsity of the superdiagonal of
  // the U_{i,j} (the first j+1 superdiagonal entries are zero).  This appears in the code below as an a > 0 check, as opposed to a sparser loop structure.
  for (int i=0;i<n;i++) // Expand variable x_i
    for (int j=0;j<degree;j++) // Multiply by U_{i,j}^{-1}
      for (int k=lambda.m-1;k>=0;k--) {
        const int a = lambda(k,i)-1-j;
        if (a > 0) // Alo -= a*A[k]
          mpz_submul_ui(A[to_flat[from_flat[k]-powers[i]]],A[k],a);
      }
}

// How many extra limbs are required to scale by d!?
static int factorial_limbs(const int d) {
  // Stirling's approximation
  const double log_factorial_d = 1-d+(d+.5)*log(d);
  return int(ceil(1/(8*sizeof(mp_limb_t)*log(2))*log_factorial_d));
}

// A specialized version of in_place_interpolating_polynomial for the univariate case.  The constant term is assumed to be zero.
// The result is scaled by degree! to avoid the need for rational arithmetic.
static void scaled_univariate_in_place_interpolating_polynomial(Subarray<mp_limb_t,2> A) {
  const int degree = A.m;
  // Multiply by the inverse of the lower triangular part.
  // Equivalently, iterate divided differences for degree passes, but skip the divisions to preserve integers.
  // Since pass p would divide by p, skipping them all multiplies the result by degree!.
  for (int pass=1;pass<=degree;pass++) {
    const int lo = max(pass-1,1);
    for (int k=degree-1;k>=lo;k--)
      mpz_sub(A[k],A[k-1]); // A[k] -= A[k-1]
  }
  // Correct for divisions we left out that weren't there, taking care to not overflow unsigned long so that we can use mpz_mul_ui
  mp_limb_t factor = 1;
  for (int k=degree-2;k>=0;k--) {
    if (factor*(k+2)/(k+2)!=factor) {
      // We're about to overflow, so multiply this factor away and reset to one
      for (int i=0;i<=k;i++)
        mpz_mul_ui(A[i],factor);
      factor = 1;
    }
    factor *= k+2;
    mpz_mul_ui(A[k],factor);
  }

  // Multiply by the inverse of the special upper triangular part.  The special upper triangular part factors into
  // bidiagonal matrices as U = U(0) U(1) ... U(d), where the superdiagonal of U(k) is k zeros followed by x(0)..x(d-k).
  // Thus, we compute U(d)^{-1} .. U(0)^{-1} A.  This process is naturally integral; no extra factors are necessary.
  // Additionally, since x(k) = k, and x(0) = 0, U(d) = 1.
  for (int k=0;k<degree;k++) // Multiply by U(k)^{-1}
    for (int i=degree-1;i>k;i--)
      mpz_submul_ui(A[i-1],A[i],i-k); // A[i-1] -= (i-k)*A[i]
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
template<class R,int m> static void assert_last_nonzero(void(*const polynomial)(R,RawArray<const Vector<Exact<1>,m>>), R result, RawArray<const Tuple<int,Vector<Quantized,m>>> X, const char* message) {
  typedef Vector<Exact<1>,m> EV;
  const int n = X.size();
  const auto Z = GEODE_RAW_ALLOCA(n,EV);
  for (const int k : range(20)) {
    for (int i=0;i<n;i++)
      Z[i] = EV(to_exact(X[i].y)+perturbation<m>(k<<10,X[i].x));
    polynomial(result,Z);
    if (last_nonzero(result)) // Even a single nonzero means we're all good
      return;
  }
  // If we reach this point, the chance of a nonzero nonmalicious polynomial is something like (1e-5)^20 = 1e-100.  Thus, we can safely assume that for the lifetime
  // of this code, we will never treat a nonzero polynomial as zero.  If this comes up, we can easily bump the threshold a bit further.
  throw AssertionError(format("%s (there is likely a bug in the calling code), X = %s",message,str(X)));
}

template<int m> bool perturbed_sign(void(*const predicate)(RawArray<mp_limb_t>,RawArray<const Vector<Exact<1>,m>>), const int degree, RawArray<const Tuple<int,Vector<Quantized,m>>> X) {
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
      const Array<mp_limb_t,2> values(lambda.m,precision,false);
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

template<int m> void perturbed_ratio(RawArray<Quantized> result, void(*const ratio)(RawArray<mp_limb_t,2>,RawArray<const Vector<Exact<1>,m>>), const int degree, RawArray<const Tuple<int,Vector<Quantized,m>>> X, const bool take_sqrt) {
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
    if (mpz_nonzero(R[r]))
      return snap_divs(result,R,take_sqrt);
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
      if (mpz_nonzero(values(j,r))) // We found a nonzero, now compute the rounded ratio
        return snap_divs(result,values[j],take_sqrt);
      else
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
      const Array<mp_limb_t,3> values(lambda.m,r+1,precision,false);
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
      int nonzero = -1;
      for (int j=0;j<lambda.m;j++)
        if (mpz_nonzero(values(j,r))) {
          if (check) // Verify that a term which used to be zero doesn't become nonzero
            GEODE_ASSERT(lambda(j,d-1));
          if (nonzero<0 || monomial_less(lambda[nonzero],lambda[j]))
            nonzero = j;
        }

      // Verify that numerator coefficients are zero for all large monomials
      for (int j=0;j<lambda.m;j++)
        if (nonzero<0 || monomial_less(lambda[nonzero],lambda[j]))
          for (int k=0;k<r;k++)
            if (mpz_nonzero(values(j,k)))
              throw OverflowError(format("perturbed_ratio: infinite result in l'Hopital expansion: %s/0",str(values(j,k))));

      // If we found a nonzero, compute the result
      if (nonzero >= 0)
        return snap_divs(result,values[nonzero],take_sqrt);

      // If we get through two levels without fixing the degeneracy, run a fast, strict identity test to make sure we weren't handed an impossible problem.
      if (d==2)
        assert_last_nonzero(ratio,values[0],X,"perturbed_ratio: identically zero denominator");
    }
  }
}

// Everything that follows is for testing purposes

static ExactInt evaluate(RawArray<const uint8_t,2> lambda, RawArray<const ExactInt> coefs, RawArray<const uint8_t> inputs) {
  GEODE_ASSERT(lambda.sizes()==vec(coefs.size(),inputs.size()));
  ExactInt sum = 0;
  for (int k=0;k<lambda.m;k++) {
    auto v = coefs[k];
    for (int i=0;i<lambda.n;i++)
      for (int j=0;j<lambda(k,i);j++)
        v *= inputs[i];
    sum += v;
  }
  return sum;
}

static void in_place_interpolating_polynomial_test(const int degree, RawArray<const uint8_t,2> lambda, RawArray<const ExactInt> coefs, const bool verbose) {
  const int precision = sizeof(ExactInt)/sizeof(mp_limb_t);
  const auto values = GEODE_RAW_ALLOCA(lambda.m*precision,mp_limb_t).reshape(lambda.m,precision);
  for (int k=0;k<lambda.m;k++)
    mpz_set(values[k],Exact<1>(evaluate(lambda,coefs,lambda[k])));
  if (verbose) {
    cout << "\ndegree = "<<degree<<"\nlambda =";
    for (int k=0;k<lambda.m;k++)
      cout << ' ' << show_monomial(lambda[k]);
    cout << "\ncoefs = "<<coefs<<"\nvalues = "<<mpz_str(values)<<endl;
  }
  const auto mcoefs = values.copy();
  in_place_interpolating_polynomial(degree,lambda,mcoefs);
  if (verbose)
    cout << "result = "<<mpz_str(mcoefs)<<endl;
  for (int k=0;k<lambda.m;k++) {
    const Exact<1> c(coefs[k]);
    GEODE_ASSERT(mcoefs[k]==asarray(c.n));
  }

  // If we're univariate, compare against the specialized routine
  if (degree+1==lambda.m) {
    const auto ucoefs = values.slice(1,degree+1).copy();
    for (int j=0;j<degree;j++)
      mpz_sub(ucoefs[j],values[0]); // ucoefs[j] -= values[0];
    scaled_univariate_in_place_interpolating_polynomial(ucoefs);
    mp_limb_t scale = 1;
    for (int k=1;k<=degree;k++)
      scale *= k;
    if (verbose)
      cout << "scale = "<<scale<<", univariate = "<<mpz_str(ucoefs)<<endl;
    for (int j=0;j<degree;j++) {
      // Compare scale*mcoefs[j+1] with ucoefs[j]
      mpz_mul_ui(mcoefs[j+1],scale);
      GEODE_ASSERT(mcoefs[j+1]==ucoefs[j]);
    }
  }
}

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
        Array<mp_limb_t> result(enough*m*power,false);
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

static void perturbed_ratio_test() {
  typedef Vector<Quantized,2> EV;
  for (const int power : vec(1,2))
    for (const int index : range(20)) {
      nasty_power = power;
      nasty_index = index;
      Vector<const Tuple<int,EV>,2> X(tuple(index,EV()),tuple(index+1,EV(perturbation<2>(7,index+1))));
      Vector<Quantized,1> result;
      perturbed_ratio(asarray(result),nasty_ratio,2+power,asarray(X),power==2);
      GEODE_ASSERT(result.x==(power==1?X.y.y.x:abs(X.y.y.x)));
    }
}

#define INSTANTIATE(m) \
  template Vector<ExactInt,m> perturbation(const int, const int); \
  template bool perturbed_sign(void(*const)(RawArray<mp_limb_t>,RawArray<const Vector<Exact<1>,m>>), const int, RawArray<const Tuple<int,Vector<Quantized,m>>>); \
  template void perturbed_ratio(RawArray<Quantized>,void(*const)(RawArray<mp_limb_t,2>, RawArray<const Vector<Exact<1>,m>>), const int, RawArray<const Tuple<int,Vector<Quantized,m>>>, bool);
INSTANTIATE(1)
INSTANTIATE(2)
INSTANTIATE(3)

}
using namespace geode;

void wrap_perturb() {
  GEODE_FUNCTION_2(perturb_monomials,monomials)
  GEODE_FUNCTION_2(perturbed_sign_test_1,perturbed_sign_test<1>)
  GEODE_FUNCTION_2(perturbed_sign_test_2,perturbed_sign_test<2>)
  GEODE_FUNCTION_2(perturbed_sign_test_3,perturbed_sign_test<3>)
  GEODE_FUNCTION(in_place_interpolating_polynomial_test)
  GEODE_FUNCTION(snap_divs_test)
  GEODE_FUNCTION(perturbed_ratio_test)
}
