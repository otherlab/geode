// General purpose black box simulation of simplicity
#pragma once

#include <other/core/exact/config.h>
#include <other/core/exact/exact.h>
#include <other/core/structure/Tuple.h>
#include <other/core/utility/IRange.h>
#include <other/core/vector/Vector.h>
#include <gmp.h>
namespace other {

// Assuming predicate(X) == 0, evaluate predicate(X+epsilon)>0 for a certain infinitesimal perturbation epsilon.  The predicate
// must be a multivariate polynomial of at most the given degree.  The permutation chosen is deterministic, independent of the
// predicate, and guaranteed to work for all possible polynomials.  Each coordinate of the X array contains (1) the value and
// (2) the index of the value, which is used to look up the fixed perturbation.
//
// Identically zero polynomials are zero regardless of perturbation; these are detected and an exception is thrown.
template<int m> OTHER_CORE_EXPORT OTHER_COLD bool perturbed_sign(exact::Exact<>(*const predicate)(RawArray<const Vector<ExactInt,m>>), const int degree, RawArray<const Tuple<int,Vector<ExactInt,m>>> X);

// Given polynomial numerator and denominator functions, evaluate numerator(X+epsilon)/denominator(X+epsilon) rounded to int for the
// same infinitesimal perturbation epsilon as in perturbed_sign.  The numerator and denominator must be multivariate polynomials of at
// most the given degree.  The rational function is packed as ratio(X) = concatenate(numerator(X),denominator(X)).  Rounding is to nearest.
//
// One key difference from perturbed_sign: the unperturbed ratio is not assumed to be singular; it will be evaluated correctly in all cases.
//
// If the rounded result does not fit into the range [-bound,bound] (e.g., if its infinite), an exception is thrown.  This should
// never happen for appropriately shielded predicates, such as constructing the intersection of two segments once perturbed_sign
// has verified that they intersect after perturbation.
//
// If take_sqrt is through, an exactly rounded square root is computed.  
template<int rp,int m> OTHER_CORE_EXPORT OTHER_COLD Vector<ExactInt,rp-1> perturbed_ratio(Vector<exact::Exact<>,rp>(*const ratio)(RawArray<const Vector<ExactInt,m>>),
                                                                                          const int degree, RawArray<const Tuple<int,Vector<ExactInt,m>>> X, const bool take_sqrt=false);

// The levelth perturbation of point i in R^m.  This is exposed for occasional special purpose use only, or as a convenient
// pseudorandom generator; normally this routine is called internally by perturbed_sign.  perturbation<m+1> starts with perturbation<m>.
template<int m> OTHER_CORE_EXPORT Vector<ExactInt,m> perturbation(const int level, const int i);

// Wrap predicate for consumption by perturbed_sign (use only via perturbed_predicate)
template<class F,int d,class... entries> static exact::Exact<> wrapped_predicate(RawArray<const Vector<ExactInt,d>> X) {
  typedef Vector<exact::Exact<1>,d> LV;
  assert(X.size()==sizeof...(entries));
  return exact::Exact<>(F::eval(LV(X[entries::value])...));
}
template<class F,int d,class... entries> OTHER_ALWAYS_INLINE static inline auto wrap_predicate(Types<entries...>)
  -> decltype(&wrapped_predicate<F,d,entries...>) {
  return &wrapped_predicate<F,d,entries...>;
}

// Given F s.t. F::eval exactly computes a polynomial in its input arguments, compute the perturbed sign of F(args).
// This is the standard way of turning an expression into a perturbed predicate.  For examples, see predicates.cpp.
template<class F,class... Args> OTHER_ALWAYS_INLINE static inline bool perturbed_predicate(const Args... args) {
  const int n = sizeof...(Args);
  const int d = First<Args...>::type::second_type::m;
  typedef Vector<exact::Exact<1>,d> LV;
  typedef decltype(F::eval(LV(args.y)...)) Result;
  const int degree = Result::degree;

  // Evaluate with integers first, hoping for a nonzero
  if (const int s = sign(F::eval(LV(args.y)...)))
    return s>0;

  // Fall back to symbolic perturbation
  const typename exact::Point<d>::type X[n] = {args...};
  return perturbed_sign(wrap_predicate<F,d>(IRange<sizeof...(Args)>()),degree,asarray(X));
}

// Compute the sign of A + sign*B*sqrt(C), where A,B,C are polynomials and |sign| = 1.  C must be positive.
namespace {
template<class A,class B,class C> struct OneSqrt { template<class... Args> static auto eval(const Args... args) // A^2 - B^2 C
  -> decltype(sqr(A::eval(args...))-sqr(B::eval(args...))*C::eval(args...)) {
  return sqr(A::eval(args...))-sqr(B::eval(args...))*C::eval(args...);
}};}
template<class A,class B,class C,class... Args> OTHER_ALWAYS_INLINE static inline bool perturbed_predicate_sqrt(const int sign, const Args... args) {
  assert(abs(sign)==1);
  assert(perturbed_predicate<C>(args...));
  // Evaluate the signs of A and B
  const int sA = perturbed_predicate<A>(args...) ? 1 : -1,
            sB = perturbed_predicate<B>(args...) ? 1 : -1;
  // If the signs match, we're done
  if (sA==sign*sB)
    return sA > 0;
  // Otherwise, we use
  //   A + sign*B*sqrt(C) > 0   iff   A > -sign*B*sqrt(C)
  //                            iff   A^2 > B^2 C     xor sA < 0
  //                            iff   A^2 - B^2 C > 0 xor sA < 0
  return perturbed_predicate<OneSqrt<A,B,C>>(args...) ^ (sA<0);
}

// Compute the sign of A + B0*sign0*sqrt(C0) + B1*sign1*sqrt(C1), where A,B0,B1,C0,C1 are polynomials and |sign0| = |sign1| = 1.  C0 and C1 must be positive.
namespace {
template<class A,class B0,class B1,class C0,class C1> struct TwoSqrtsAlpha { template<class... Args> static auto eval(const Args... args)
  -> decltype(sqr(A::eval(args...))) {
  return sqr(A::eval(args...)) + sqr(B0::eval(args...))*C0::eval(args...)
                               - sqr(B1::eval(args...))*C1::eval(args...);
}};
template<class A,class B0> struct TwoSqrtsBeta { template<class... Args> static auto eval(const Args... args)
  -> decltype(A::eval(args...)*B0::eval(args...)) {
  return small_mul(2,A::eval(args...)*B0::eval(args...));
}};}
template<class A,class B0,class B1,class C0,class C1,class... Args> OTHER_ALWAYS_INLINE static inline bool perturbed_predicate_two_sqrts(const int sign0, const int sign1, const Args... args) {
  assert(abs(sign0)==1 && abs(sign1)==1);
  assert(perturbed_predicate<C0>(args...));
  assert(perturbed_predicate<C1>(args...));
  // Let
  //   G = A + B0*sign0*sqrt(C0)
  // Then our predicate is
  //   G + sign1*B1*sqrt(C1)
  // First, we try for early return:
  const int sA = perturbed_predicate<A>(args...)   ? 1 : -1,
            sB0 = perturbed_predicate<B0>(args...) ? 1 : -1,
            sB1 = perturbed_predicate<B1>(args...) ? 1 : -1,
            sG = sA==sign0*sB0 ? sA
                               : perturbed_predicate<OneSqrt<A,B0,C0>>(args...) ^ (sA<0) ? 1 : -1;
  if (sG==sign1*sB1)
    return sG > 0;
  // Otherwise, we use
  //   G + sign1*B1*sqrt(C1) > 0   iff   G > -sign1*B1*sqrt(C1)
  //                               iff   A + B0*sign0*sqrt(C0) > -sign1*B1*sqrt(C1)
  //                               iff   A^2 + 2*A*B0*sign0*sqrt(C0) + B0^2*C0 > B1^2*C1     xor sG < 0
  //                               iff   A^2 + B0^2*C0 - B1^2*C1 > -2*A*B0*sign0*sqrt(C0)    xor sG < 0
  //                               iff   Alpha + Beta*sign0*sqrt(C0) > 0                     xor sG < 0
  // where
  //   Alpha = A^2 + B0^2*C0 - B1^2*C1
  //   Beta = 2*A*B0
  // Here are some more signs:
  typedef TwoSqrtsAlpha<A,B0,B1,C0,C1> Alpha;
  typedef TwoSqrtsBeta<A,B0> Beta;
  const int sAlpha = perturbed_predicate<Alpha>(args...) ? 1 : -1,
            sBeta = sign0*sA*sB0;
  if (sAlpha==sBeta)
    return (sAlpha > 0) ^ (sG < 0);
  // Otherwise, evaluate an even larger predicate.  For completeness, this larger predicate has the form (ignoring sign flips)
  //   G + sign1*B1*sqrt(C1) > 0   iff   A^2 + B0^2*C0 - B1^2*C1 > -2*A*B0*sign0*sqrt(C0) 
  //                               iff   (A^2 + B0^2*C0 - B1^2*C1)^2 > 4*A^2*B0^2*C0
  //                               iff   A^4 + B0^4*C0^2 + B1^4*C1^2 + 2*A^2*B0^2*C0 - 2*A^2*B1^2*C1 - 2*B0^2*B1^2*C0*C1 > 4*A^2*B0^2*C0
  //                               iff   A^4 + B0^4 C0^2 + B1^4 C1^2 - 2 A^2 B0^2 C0 - 2 A^2 B1^2 C1 - 2 B0^2 B1^2 C0 C1 > 0
  return perturbed_predicate<OneSqrt<Alpha,Beta,C0>>(args...) ^ (sAlpha < 0) ^ (sG < 0);
}

}
