// Operations on polynomials for use in symbolic perturbation

#include <geode/exact/polynomial.h>
#include <geode/array/alloca.h>
namespace geode {

using std::cout;
using std::endl;

Array<const uint8_t,2> monomials(const int degree, const int variables) {
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

void in_place_interpolating_polynomial(const int degree, RawArray<const uint8_t,2> lambda, Subarray<mp_limb_t,2> A) {
  // For now we are lazy, and index using a rectangular helper array mapping multi-indices to flat indices
  const int n = lambda.n;
  Array<int> powers(n+1,uninit);
  powers[0] = 1;
  for (int i=0;i<n;i++)
    powers[i+1] = powers[i]*(degree+1);
  Array<uint16_t> to_flat(powers.back(),uninit);
  Array<int> from_flat(lambda.m,uninit);
  to_flat.fill(-1);
  for (int k=0;k<lambda.m;k++) {
    int f = 0;
    for (int i=0;i<n;i++)
      f += powers[i]*lambda(k,i);
    from_flat[k] = f;
    to_flat[f] = k;
  }

  // Bookkeeping information for the divided difference algorithm
  Array<Vector<int,2>> info(lambda.m,uninit); // m,alpha[m] for each tick
  for (int k=0;k<lambda.m;k++)
    info[k].set(0,lambda(k,0));

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
    }
    next_pass:;
  }

  // At this point A contains the coefficients of the interpolating polynomial in the Newton basis, and we are halfway 
  // expand the Newton basis out into the monomial basis.  We start with A(beta) = g(beta), the coefficient of the Newt
  // We seek h(alpha) for alpha <= beta, the coefficients of the monomial basis polynomials x^alpha.  This gives rise t
  // matrix M = M_{alpha,beta}:
  //
  //   h = M g
  //   M_ab = 0 unless a <= b componentwise: M is upper triangular w.r.t. the partial order on ticks
  //   M_bb = 1: M is special upper triangular, since Newton basis polynomials are monic
  //
  // Since the multivariate Newton polynomials are simple products of univariate Newton polynomials for each variable, 
  // block diagonal matrices M_i for each variable x_i, corresponding to expanding one variable while leaving the other
  // factors into a product of special upper bidiagonal matrices U_{i,j}^{-1}, as given in the univariate case below.
  //
  // Note: Since lambda is not ordered with respect to any single variable, it is difficult to take advantage of the sp
  // the U_{i,j} (the first j+1 superdiagonal entries are zero).  This appears in the code below as an a > 0 check, as 
  for (int i=0;i<n;i++) // Expand variable x_i
    for (int j=0;j<degree;j++) // Multiply by U_{i,j}^{-1}
      for (int k=lambda.m-1;k>=0;k--) {
        const int a = lambda(k,i)-1-j;
        if (a > 0) // Alo -= a*A[k]
          mpz_submul_ui(A[to_flat[from_flat[k]-powers[i]]],A[k],a);
      }
}

void scaled_univariate_in_place_interpolating_polynomial(Subarray<mp_limb_t,2> A) {
  const int degree = A.m;
  // Multiply by the inverse of the lower triangular part.
  // Equivalently, iterate divided differences for degree passes, but skip the divisions to preserve integers.
  // Since pass p would divide by p, skipping them all multiplies the result by degree!.
  for (int pass=1;pass<=degree;pass++) {
    const int lo = max(pass-1,1);
    for (int k=degree-1;k>=lo;k--)
      mpz_sub(A[k],A[k-1]); // A[k] -= A[k-1]
  }
  // Correct for divisions we left out that weren't there, taking care to not overflow unsigned long so that we can use
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
  // bidiagonal matrices as U = U(0) U(1) ... U(d), where the superdiagonal of U(k) is k zeros followed by x(0)..x(d-k)
  // Thus, we compute U(d)^{-1} .. U(0)^{-1} A.  This process is naturally integral; no extra factors are necessary.
  // Additionally, since x(k) = k, and x(0) = 0, U(d) = 1.
  for (int k=0;k<degree;k++) // Multiply by U(k)^{-1}
    for (int i=degree-1;i>k;i--)
      mpz_submul_ui(A[i-1],A[i],i-k); // A[i-1] -= (i-k)*A[i]
}

// Everything that follows is for testing purposes

static ExactInt evaluate(RawArray<const uint8_t,2> lambda, RawArray<const ExactInt> coefs,
                         RawArray<const uint8_t> inputs) {
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

void in_place_interpolating_polynomial_test(const int degree, RawArray<const uint8_t,2> lambda, 
                                            RawArray<const ExactInt> coefs, const bool verbose) {
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

}
