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
template<int m> OTHER_CORE_EXPORT OTHER_COLD bool perturbed_sign(exact::Exact<>(*const predicate)(RawArray<const Vector<exact::Int,m>>), const int degree, RawArray<const Tuple<int,Vector<exact::Int,m>>> X);

// Given polynomial numerator and denominator functions, evaluate numerator(X+epsilon)/denominator(X+epsilon) rounded to int for the
// same infinitesimal perturbation epsilon as in perturbed_sign.  The numerator and denominator must be multivariate polynomials of at
// most the given degree.  The rational function is packed as ratio(X) = concatenate(numerator(X),denominator(X)).  Rounding is to nearest.
//
// One key difference from perturbed_sign: the unperturbed ratio is not assumed to be singular; it will be evaluated correctly in all cases.
//
// If the rounded result does not fit into the range [-bound,bound] (e.g., if its infinite), an exception is thrown.  This should
// never happen for appropriately shielded predicates, such as constructing the intersection of two segments once perturbed_sign
// has verified that they intersect after perturbation.
template<int m> OTHER_CORE_EXPORT OTHER_COLD Vector<exact::Int,m> perturbed_ratio(Vector<exact::Exact<>,m+1>(*const ratio)(RawArray<const Vector<exact::Int,m>>), const int degree, RawArray<const Tuple<int,Vector<exact::Int,m>>> X);

// The levelth perturbation of point i in R^m.  This is exposed for occasional special purpose use only, or as a convenient
// pseudorandom generator; normally this routine is called internally by perturbed_sign.
template<int m> OTHER_CORE_EXPORT Vector<exact::Int,m> perturbation(const int level, const int i);

// Wrap predicate for consumption by perturbed_sign (use only via perturbed_predicate)
template<class F,int d,class... entries> static exact::Exact<> wrapped_predicate(RawArray<const Vector<exact::Int,d>> X) {
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

}
