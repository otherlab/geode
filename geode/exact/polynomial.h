// Operations on polynomials for use in symbolic perturbation

#include <geode/exact/Exact.h>
#include <geode/array/Array2d.h>
namespace geode {

// List all n-variate monomials of degree <= d, ordered by ascending total degree and then arbitrarily.
// Warning: This is the order needed for divided difference interpolation, but is *not* the correct infinitesimal
// size order.
Array<const uint8_t,2> monomials(const int degree, const int variables);

// Pretty 

// The relative size ordering on infinitesimals
static inline bool monomial_less(RawArray<const uint8_t> a, RawArray<const uint8_t> b) {
  assert(a.size()==b.size());
  for (int i=a.size()-1;i>=0;i--)
    if (a[i] != b[i])
      return a[i]>b[i];
  return false;
}

// Given the values of a polynomial at every point in the standard "easy corner", solve for the monomial
// coefficients using divided differences.  lambda and A are as in Neidinger.  We assume lambda is partially
// sorted by total degree.
void in_place_interpolating_polynomial(const int degree, RawArray<const uint8_t,2> lambda, Subarray<mp_limb_t,2> A);

// How many extra limbs are required to scale by d!?
static inline int factorial_limbs(const int d) {
  // Stirling's approximation
  const double log_factorial_d = 1-d+(d+.5)*log(d);
  return int(ceil(1/(8*sizeof(mp_limb_t)*log(2))*log_factorial_d));
}

// A specialized version of in_place_interpolating_polynomial for the univariate case.  The constant term is
// assumed to be zero.  The result is scaled by degree! to avoid the need for rational arithmetic.
void scaled_univariate_in_place_interpolating_polynomial(Subarray<mp_limb_t,2> A);

// For testing purposes
void in_place_interpolating_polynomial_test(const int degree, RawArray<const uint8_t,2> lambda, 
                                            RawArray<const ExactInt> coefs, const bool verbose);

}
