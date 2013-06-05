// General purpose black box simulation of simplicity

#include <other/core/exact/perturb.h>
#include <other/core/exact/math.h>
#include <other/core/array/alloca.h>
#include <other/core/array/Array2d.h>
#include <other/core/python/wrap.h>
#include <other/core/random/counter.h>
#include <other/core/utility/move.h>
#include <other/core/vector/Matrix.h>
namespace other {

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
using exact::Exact;
using exact::init_set_steal;

// Compile time debugging support
static const bool check = false;
static const bool verbose = false;

// Our fixed deterministic pseudorandom perturbation sequence.  We limit ourselves to 32 bits so that we can pull four values out of a uint128_t.
template<int m> inline Vector<ExactInt,m> perturbation(const int level, const int i) {
  BOOST_STATIC_ASSERT(m<=4);
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
  OTHER_ASSERT(num <= (1<<20));
  Array<uint8_t,2> results(num,variables);

  // We simulate a stack manually to avoid recursion
  if (variables) {
    int next = 0;
    const auto alpha = OTHER_RAW_ALLOCA(variables,uint8_t);
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

// Faster division of rationals by ints
static inline void mpq_div_si(mpq_t x, long n) {
  // Make n relatively prime to numerator(x)
  const auto gcd = mpz_gcd_ui(0,mpq_numref(x),abs(n));
  n /= gcd;
  mpz_divexact_ui(mpq_numref(x),mpq_numref(x),gcd);
  // Perform the division
  mpz_mul_si(mpq_denref(x),mpq_denref(x),n);
}

namespace {
template<class G> struct Destroyer {
  RawArray<G> values;
  Destroyer(RawArray<G> values) : values(values) {}
  ~Destroyer() { for (auto& x : values) clear(&x); }

  void clear(mpz_t x) { mpz_clear(x); }
  void clear(mpq_t x) { mpq_clear(x); }
};
}

// Given the values of a polynomial at every point in the standard "easy corner", solve for the monomial coefficients using divided differences
// lambda and A are as in Neidinger.  We assume lambda is partially sorted by total degree.
static void in_place_interpolating_polynomial(const int degree, const RawArray<const uint8_t,2> lambda, RawArray<__mpq_struct> A) {
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
      mpq_sub(&A[k],&A[k],&A[child]); // A[k] -= A[child]
      mpq_div_si(&A[k],lambda(k,I.x)-I.y); // A[k] /= lambda(k,I.x)-I.y
      // In self check mode, verify that the necessary f[alpha,beta] values were available
      if (check) {
        alpha(k,I.x)--;
        OTHER_ASSERT(alpha[k]==alpha[child]);
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
  mpq_t tmp;
  mpq_init(tmp);
  for (int i=0;i<n;i++) // Expand variable x_i
    for (int j=0;j<degree;j++) // Multiply by U_{i,j}^{-1}
      for (int k=lambda.m-1;k>=0;k--) {
        const int a = lambda(k,i)-1-j;
        if (a > 0) { // Alo -= a*A[k]
          mpq_set_si(tmp,a,1);
          mpq_mul(tmp,tmp,&A[k]);
          auto& Alo = A[to_flat[from_flat[k]-powers[i]]];
          mpq_sub(&Alo,&Alo,tmp);
        }
      }
  mpq_clear(tmp);
}

// A specialized version of in_place_interpolating_polynomial for the univariate case.  The constant term is assumed to be zero.
// The result is scaled by degree! to avoid the need for rational arithmetic.
static void scaled_univariate_in_place_interpolating_polynomial(const int degree, RawArray<__mpz_struct> A) {
  assert(degree==A.size());
  // Multiply by the inverse of the lower triangular part.
  // Equivalently, iterate divided differences for degree passes, but skip the divisions to preserve integers.
  // Since pass p would divide by p, skipping them all multiplies the result by degree!.
  for (int pass=1;pass<=degree;pass++) {
    const int lo = max(pass-1,1);
    for (int k=degree-1;k>=lo;k--)
      mpz_sub(&A[k],&A[k],&A[k-1]); // A[k] -= A[k-1]
  }
  // Correct for divisions we left out that weren't there, taking care to not overflow unsigned long so that we can use mpz_mul_ui
  unsigned long factor = 1;
  for (int k=degree-2;k>=0;k--) {
    if (factor*(k+2)/(k+2)!=factor) {
      // We're about to overflow, so multiply this factor away and reset to one
      for (int i=0;i<=k;i++)
        mpz_mul_ui(&A[i],&A[i],factor);
      factor = 1;
    }
    factor *= k+2;
    mpz_mul_ui(&A[k],&A[k],factor);
  }

  // Multiply by the inverse of the special upper triangular part.  The special upper triangular part factors into
  // bidiagonal matrices as U = U(0) U(1) ... U(d), where the superdiagonal of U(k) is k zeros followed by x(0)..x(d-k).
  // Thus, we compute U(d)^{-1} .. U(0)^{-1} A.  This process is naturally integral; no extra factors are necessary.
  // Additionally, since x(k) = k, and x(0) = 0, U(d) = 1.
  for (int k=0;k<degree;k++) // Multiply by U(k)^{-1}
    for (int i=degree-1;i>k;i--)
      mpz_submul_ui(&A[i-1],&A[i],i-k); // A[i-1] -= (i-k)*A[i]
}

static bool last_nonzero(const Exact<>& x) {
  return sign(x)!=0;
}

template<int m> static bool last_nonzero(const Vector<Exact<>,m>& x) {
  return sign(x.back())!=0;
}

// Check for identically zero polynomials using randomized polynomial identity testing
template<class R,int m> static void assert_last_nonzero(R(*const polynomial)(RawArray<const Vector<ExactInt,m>>), RawArray<const Tuple<int,Vector<ExactInt,m>>> X, const char* message) {
  const int n = X.size();
  const auto Z = OTHER_RAW_ALLOCA(n,Vector<ExactInt,m>);
  for (const int k : range(20)) {
    for (int i=0;i<n;i++)
      Z[i] = X[i].y+perturbation<m>(k<<10,X[i].x);
    if (last_nonzero(polynomial(Z))) // Even a single nonzero means we're all good
      return;
  }
  // If we reach this point, the chance of a nonzero nonmalicious polynomial is something like (1e-5)^20 = 1e-100.  Thus, we can safely assume that for the lifetime
  // of this code, we will never treat a nonzero polynomial as zero.  If this comes up, we can easily bump the threshold a bit further.
  throw AssertionError(format("%s (there is likely a bug in the calling code), X = %s",message,str(X)));
}

template<int m> bool perturbed_sign(Exact<>(*const predicate)(RawArray<const Vector<ExactInt,m>>), const int degree, RawArray<const Tuple<int,Vector<ExactInt,m>>> X) {
  if (check)
    OTHER_WARNING("Expensive consistency checking enabled");

  const int n = X.size();
  if (verbose)
    cout << "perturbed_sign:\n  degree = "<<degree<<"\n  X = "<<X<<endl;

  // If desired, verify that predicate(X) == 0
  const auto Z = OTHER_RAW_ALLOCA(n,Vector<ExactInt,m>);
  if (check) {
    for (int i=0;i<n;i++)
      Z[i] = X[i].y;
    OTHER_ASSERT(!sign(predicate(Z)));
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
    const auto values = OTHER_RAW_ALLOCA(degree,__mpz_struct);
    for (int j=0;j<degree;j++) {
      for (int i=0;i<n;i++)
        Z[i] = X[i].y+(j+1)*Y[i];
      init_set_steal(&values[j],predicate(Z).n);
      if (verbose)
        cout << "  predicate("<<Z<<") = "<<values[j]<<endl;
    }
    Destroyer<__mpz_struct> destroyer(values);

    // Find an interpolating polynomial, overriding the input with the result.
    scaled_univariate_in_place_interpolating_polynomial(degree,values);
    if (verbose)
      cout << "  coefs = "<<values<<endl;

    // Compute sign
    for (int j=0;j<degree;j++)
      if (const int sign = mpz_sgn(&values[j]))
        return sign>0;
  }

  {
    // Add one perturbation level after another until we hit a nonzero polynomial.  Our current implementation duplicates
    // work from one iteration to the next for simplicity, which is fine since the first interation suffices almost always.
    Array<__mpq_struct> values;
    for (int d=2;;d++) {
      // Compute the next level of perturbations
      Y.resize(d*n);
      for (int i=0;i<n;i++)
        Y[(d-1)*n+i] = perturbation<m>(d,X[i].x);

      // Evaluate polynomial at every point in an "easy corner"
      const auto lambda = monomials(degree,d);
      values.resize(lambda.m,false,false);
      for (int j=0;j<lambda.m;j++) {
        for (int i=0;i<n;i++)
          Z[i] = X[i].y+lambda(j,0)*Y[i];
        for (int v=1;v<d;v++)
          for (int i=0;i<n;i++)
            Z[i] += lambda(j,v)*Y[v*n+i];
        init_set_steal(mpq_numref(&values[j]),predicate(Z).n);
        mpz_init_set_ui(mpq_denref(&values[j]),1);
      }
      Destroyer<__mpq_struct> destroyer(values);

      // Find an interpolating polynomial, overriding the input with the result.
      in_place_interpolating_polynomial(degree,lambda,values);

      // Compute sign
      int sign = 0;
      int sign_j = -1;
      for (int j=0;j<lambda.m;j++)
        if (const int s = mpq_sgn(&values[j])) {
          if (check) // Verify that a term which used to be zero doesn't become nonzero
            OTHER_ASSERT(lambda(j,d-1));
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
        assert_last_nonzero(predicate,X,"perturbed_sign: identically zero predicate");
    }
  }
}

// Convert to an ExactInt or return numeric_limits<ExactInt>::max()/2 on overflow.  We do not compare against exact::bound.
static ExactInt mpz_get_exact_int(mpz_t x) {
  BOOST_STATIC_ASSERT(sizeof(ExactInt)<=sizeof(long) || sizeof(long)==4); // Check that we're on a 32-bit or 64-bit machine
  const auto overflow = numeric_limits<ExactInt>::max()/2;
  if (sizeof(ExactInt)==sizeof(long))
    return mpz_fits_slong_p(x) ? ExactInt(mpz_get_si(x))
                               : overflow;
  else {
    if (!mpz_sgn(x))
      return 0;
    if (mpz_sizeinbase(x,2) > 8*sizeof(ExactInt)-2)
      return overflow;
    ExactInt n;
    mpz_export(&n,0,-1,sizeof(ExactInt),0,0,x);
    return n;
  }
}

// Cast num/den to an int, rounding towards nearest.  num is destroyed.
static ExactInt snap_div(mpz_t num, mpz_t den, const bool take_sqrt) {
  if (!take_sqrt) {
    mpz_mul_2exp(num,num,1); // num *= 2
    mpz_tdiv_q(num,num,den); // num /= den, rounding towards zero
    const auto two_ratio = mpz_get_exact_int(num),
               ratio = (two_ratio>0?1:-1)*((1+abs(two_ratio))>>1); // Divide by two, rounding odd numbers away from zero
    if (abs(ratio) <= exact::bound)
      return ratio;
    throw OverflowError(format("perturbed_ratio: overflow in l'Hopital expansion: abs(%s/2) > %d",str(num),exact::bound));
  } else {
    if (mpz_sgn(num) && mpz_sgn(num)!=mpz_sgn(den))
      throw RuntimeError("perturbed_ratio: negative value in square root");
    // We seek a nonnegative integer x s.t.
    //   x-1/2 <= sqrt(num/den) < x+1/2
    //   2x-1 <= sqrt(4num/den) < 2x+1
    // Now the leftmost and rightmost expressions are integral, so we can take floors to get
    //   2x-1 <= floor(sqrt(4num/den)) < 2x+1
    // Since sqrt is monotonic and maps integers to integers, floor(sqrt(floor(x))) = floor(sqrt(x)), so
    //   2x-1 <= floor(sqrt(floor(4num/den))) < 2x+1
    //   2x <= 1+floor(sqrt(floor(4num/den))) < 2x+2
    //   x <= (1+floor(sqrt(floor(4num/den))))//2 < x+1
    //   x = (1+floor(sqrt(floor(4num/den))))//2
    mpz_mul_2exp(num,num,2); // 4num
    mpz_tdiv_q(num,num,den); // floor(4num/den)
    mpz_sqrt(num,num); // floor(sqrt(floor(4num/den)))
    const auto x = (1+mpz_get_exact_int(num))/2;
    if (x <= exact::bound)
      return x;
    throw OverflowError("perturbed_ratio: overflow in l'Hopital expansion with sqrt");
  }
}

template<int rp,int m> Vector<ExactInt,rp-1> perturbed_ratio(Vector<exact::Exact<>,rp>(*const ratio)(RawArray<const Vector<ExactInt,m>>), const int degree, RawArray<const Tuple<int,Vector<ExactInt,m>>> X, const bool take_sqrt) {
  const int r = rp-1;
  const int n = X.size();

  // Check if the ratio is nonsingular before perturbation
  const auto Z = OTHER_RAW_ALLOCA(n,Vector<ExactInt,m>);
  {
    for (int i=0;i<n;i++)
      Z[i] = X[i].y;
    auto R = ratio(Z);
    if (mpz_sgn(R[r].n)) {
      Vector<ExactInt,r> result;
      for (int k=0;k<r;k++)
        result[k] = snap_div(R[k].n,R[r].n,take_sqrt);
      return result;
    }
  }

  // Check the first perturbation level with specialized code
  vector<Vector<ExactInt,m>> Y(n); // perturbations
  {
    // Compute the first level of perturbations
    for (int i=0;i<n;i++)
      Y[i] = perturbation<m>(1,X[i].x);

    // Evaluate polynomial at epsilon = 1, ..., degree
    const auto values = OTHER_RAW_ALLOCA((r+1)*degree,__mpz_struct).reshape(r+1,degree);
    for (int j=0;j<degree;j++) {
      for (int i=0;i<n;i++)
        Z[i] = X[i].y+(j+1)*Y[i];
      auto R = ratio(Z);
      for (int k=0;k<=r;k++)
        init_set_steal(&values(k,j),R[k].n);
    }
    Destroyer<__mpz_struct> destroyer(values.flat);

    // Find interpolating polynomials, overriding the input with the result.
    for (int k=0;k<=r;k++)
      scaled_univariate_in_place_interpolating_polynomial(degree,values[k]);

    // Find the largest (lowest degree) nonzero denominator coefficient.  If we detect an infinity during this process, explode.
    for (int j=0;j<degree;j++) {
      auto& den = values(r,j);
      if (mpz_sgn(&den)) {
        // We found a nonzero, now compute the rounded ratio
        Vector<ExactInt,r> result;
        for (int k=0;k<r;k++)
          result[k] = snap_div(&values(k,j),&den,take_sqrt);
        return result;
      } else
        for (int k=0;k<r;k++)
          if (mpz_sgn(&values(k,j)))
            throw OverflowError(format("perturbed_ratio: infinite result in l'Hopital expansion: %s/0",str(&values(k,j))));
    }
  }

  {
    // Add one perturbation level after another until we hit a nonzero denominator.  Our current implementation duplicates
    // work from one iteration to the next for simplicity, which is fine since the first interation suffices almost always.
    Array<__mpq_struct,2> values;
    for (int d=2;;d++) {
      // Compute the next level of perturbations
      Y.resize(d*n);
      for (int i=0;i<n;i++)
        Y[(d-1)*n+i] = perturbation<m>(d,X[i].x);

      // Evaluate polynomial at every point in an "easy corner"
      const auto lambda = monomials(degree,d);
      values.resize(r+1,lambda.m,false,false);
      for (int j=0;j<lambda.m;j++) {
        for (int i=0;i<n;i++)
          Z[i] = X[i].y+lambda(j,0)*Y[i];
        for (int v=1;v<d;v++)
          for (int i=0;i<n;i++)
            Z[i] += lambda(j,v)*Y[v*n+i];
        auto R = ratio(Z);
        for (int k=0;k<=r;k++) {
          init_set_steal(mpq_numref(&values(k,j)),R[k].n);
          mpz_init_set_ui(mpq_denref(&values(k,j)),1);
        }
      }
      Destroyer<__mpq_struct> destroyer(values.flat);

      // Find interpolating polynomials, overriding the input with the result.
      for (int k=0;k<=r;k++)
        in_place_interpolating_polynomial(degree,lambda,values[k]);

      // Find the largest nonzero denominator coefficient
      int nonzero = -1;
      for (int j=0;j<lambda.m;j++)
        if (mpq_sgn(&values(r,j))) {
          if (check) // Verify that a term which used to be zero doesn't become nonzero
            OTHER_ASSERT(lambda(j,d-1));
          if (nonzero<0 || monomial_less(lambda[nonzero],lambda[j]))
            nonzero = j;
        }

      // Verify that numerator coefficients are zero for all large monomials
      for (int j=0;j<lambda.m;j++)
        if (nonzero<0 || monomial_less(lambda[nonzero],lambda[j]))
          for (int k=0;k<r;k++)
            if (mpq_sgn(&values(k,j)))
              throw OverflowError(format("perturbed_ratio: infinite result in l'Hopital expansion: %s/0",str(&values(k,j))));

      // If we found a nonzero, compute the result
      if (nonzero >= 0) {
        auto& den = values(r,nonzero);
        Vector<ExactInt,r> result;
        for (int k=0;k<r;k++) {
          auto& num = values(k,nonzero);
          mpq_div(&num,&num,&den); // num /= den 
          result[k] = snap_div(mpq_numref(&num),mpq_denref(&num),take_sqrt);
        }
        return result;
      }

      // If we get through two levels without fixing the degeneracy, run a fast, strict identity test to make sure we weren't handed an impossible problem.
      if (d==2)
        assert_last_nonzero(ratio,X,"perturbed_ratio: identically zero denominator");
    }
  }
}

// Everything that follows is for testing purposes

static ExactInt evaluate(RawArray<const uint8_t,2> lambda, RawArray<const ExactInt> coefs, RawArray<const uint8_t> inputs) {
  OTHER_ASSERT(lambda.sizes()==vec(coefs.size(),inputs.size()));
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
  Array<__mpz_struct> values_z(lambda.m,false);
  Array<__mpq_struct> values_q(lambda.m,false);
  for (int k=0;k<lambda.m;k++) {
    init_set_steal(&values_z[k],evaluate(lambda,coefs,lambda[k]));
    mpq_init(&values_q[k]);
    mpz_set(mpq_numref(&values_q[k]),&values_z[k]);
  }
  if (verbose) {
    cout << "\ndegree = "<<degree<<"\nlambda =";
    for (int k=0;k<lambda.m;k++)
      cout << ' ' << show_monomial(lambda[k]);
    cout << "\ncoefs = "<<coefs<<"\nvalues = "<<values_z<<endl;
  }
  in_place_interpolating_polynomial(degree,lambda,values_q);
  if (verbose)
    cout << "result = "<<values_q<<endl;
  for (int k=0;k<lambda.m;k++)
    OTHER_ASSERT(!mpq_cmp_si(&values_q[k],coefs[k],1));

  // If we're univariate, compare against the specialized routine
  if (degree+1==lambda.m) {
    for (int j=1;j<=degree;j++)
      mpz_sub(&values_z[j],&values_z[j],&values_z[0]); // values_z[j] -= values_z[0];
    scaled_univariate_in_place_interpolating_polynomial(degree,values_z.slice(1,degree+1));
    unsigned long scale = 1;
    for (int k=1;k<=degree;k++)
      scale *= k;
    mpz_mul_ui(&values_z[0],&values_z[0],scale); // values_z[0] *= scale;
    if (verbose)
      cout << "scale = "<<scale<<", univariate = "<<values_z<<endl;
    for (int k=0;k<lambda.m;k++) {
      // Compare scale*values_q[k] with values_z[k]
      __mpq_struct* vq = &values_q[k];
      mpz_mul_ui(mpq_numref(vq),mpq_numref(vq),scale);
      mpq_canonicalize(vq);
      OTHER_ASSERT(!mpz_cmp(mpq_numref(vq),&values_z[k]) && !mpz_cmp_ui(mpq_denref(vq),1));
    }
  }

  // Free memory
  for (auto& v : values_z)
    mpz_clear(&v);
  for (auto& v : values_q)
    mpq_clear(&v);
}

// Test against malicious predicates that are zero along 0, 1, or 2 perturbation levels.

static int nasty_index, nasty_degree;

static Exact<> nasty_pow(Exact<>&& x) {
  switch (nasty_degree) {
    case 1: return other::move(x);
    case 2: return sqr(other::move(x));
    case 3: return cube(other::move(x));
    default: OTHER_FATAL_ERROR();
  };
}

template<class In> static Exact<> nasty_predicate(RawArray<const Vector<In,1>> X) {
  return nasty_pow(Exact<>(X[0].x));
}

template<class In> static Exact<> nasty_predicate(RawArray<const Vector<In,2>> X) {
  typedef Vector<Exact<>,2> EV;
  return nasty_pow(edet(EV(X[0]),
                        EV(perturbation<2>(1,nasty_index))));
}

template<class In> static Exact<> nasty_predicate(RawArray<const Vector<In,3>> X) {
  typedef Vector<Exact<>,3> EV;
  return nasty_pow(edet(EV(X[0]),
                        EV(perturbation<3>(1,nasty_index)),
                        EV(perturbation<3>(2,nasty_index))));
}

template<int m> static void perturbed_sign_test() {
  for (const int degree : vec(1,2,3))
    for (const int index : range(20)) {
      // Evaluate perturbed sign using our fancy routine
      nasty_degree = degree;
      nasty_index = index;
      Array<Tuple<int,Vector<ExactInt,m>>> fX(1);
      fX[0].x = index;
      const bool fast = perturbed_sign<m>(nasty_predicate<ExactInt>,degree,fX);
      OTHER_ASSERT((degree&1) || fast);
      // Evaluate the series out to several terms using brute force
      Vector<Exact<>,m> sX;
      Array<int> powers(m+1); // Choose powers of 2 to approximate nested infinitesimals
      for (int i=0;i<m;i++)
        powers[i+1] = (degree+1)*powers[i]+128;
      mpz_t yp;
      mpz_init(yp);
      for (int i=0;i<=m+1;i++) {
        if (i) {
          const auto y = perturbation<m>(i,index);
          for (int j=0;j<m;j++) {
            auto& x = sX[j];
            mpz_set_si(yp,y[j]);
            mpz_mul_2exp(yp,yp,powers.back()-powers[i-1]); // yp = y[j]<<(powers[-1]-powers[i-1])
            mpz_add(x.n,x.n,yp); // x += yp
          }
        }
        // We should be initially zero, and then match the correct sign once nonzero
        const int slow = sign(nasty_predicate<Exact<>>(RawArray<const Vector<Exact<>,m>>(1,&sX)));
        if (0) {
          cout << "m "<<m<<", degree "<<degree<<", index "<<index<<", i "<<i<<", fast "<<2*fast-1<<", slow "<<slow<<endl;
          cout << "  fX = "<<fX[0]<<", sX = "<<sX<<endl;
        }
        OTHER_ASSERT(slow==(i<m?0:2*fast-1));
      }
      mpz_clear(yp);
    }
}

// Note: Since perturb_ratio requires the result to fit inside [-exact::bound,exact::bound], it is hard to construct
// an interesting test function that doesn't have geometric meaning.  Therefore, we don't try, and instead rely on
// the construction-specific tests in constructions.cpp.

#define INSTANTIATE_RATIO(r,m) \
  template Vector<ExactInt,r> perturbed_ratio(Vector<Exact<>,r+1>(*const)(RawArray<const Vector<ExactInt,m>>), const int, RawArray<const Tuple<int,Vector<ExactInt,m>>>, bool);
#define INSTANTIATE(m) \
  template Vector<ExactInt,m> perturbation(const int, const int); \
  template bool perturbed_sign(Exact<>(*const)(RawArray<const Vector<ExactInt,m>>), const int, RawArray<const Tuple<int,Vector<ExactInt,m>>>); \
  INSTANTIATE_RATIO(m,m)
INSTANTIATE(1)
INSTANTIATE(2)
INSTANTIATE(3)
INSTANTIATE_RATIO(2,3)

}
using namespace other;

void wrap_perturb() {
  OTHER_FUNCTION_2(perturb_monomials,monomials)
  OTHER_FUNCTION_2(perturbed_sign_test_1,perturbed_sign_test<1>)
  OTHER_FUNCTION_2(perturbed_sign_test_2,perturbed_sign_test<2>)
  OTHER_FUNCTION_2(perturbed_sign_test_3,perturbed_sign_test<3>)
  OTHER_FUNCTION(in_place_interpolating_polynomial_test)
}
