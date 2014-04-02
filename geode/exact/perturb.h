// General purpose black box simulation of simplicity
#pragma once

#include <geode/exact/config.h>
#include <geode/exact/debug.h>
#include <geode/exact/Exact.h>
#include <geode/exact/Interval.h>
#include <geode/exact/irreducible.h>
#include <geode/structure/Tuple.h>
#include <geode/utility/IRange.h>
#include <geode/vector/Vector.h>
namespace geode {

// sys/termios.h on Mac was defining B0 as a macro.  Don't.
#undef B0

// Evaluate predicate(X+epsilon)>0 for a certain infinitesimal perturbation epsilon.  The predicate must be a
// multivariate polynomial of at most the given degree.  The permutation chosen is deterministic, independent
// of the predicate, and guaranteed to work for all possible polynomials.  Each coordinate of the X array contains
// (1) the value and (2) the index of the value, which is used to look up the fixed perturbation.
//
// Identically zero polynomials are zero regardless of perturbation; these are detected and an exception is thrown.
// predicate should compute a quantity of type Exact<degree>, then copy it into result with mpz_set.
template<class PerturbedT> GEODE_CORE_EXPORT GEODE_COLD bool
perturbed_sign(void(*const predicate)(RawArray<mp_limb_t>,RawArray<const Vector<Exact<1>,PerturbedT::m>>),
               const int degree, RawArray<const PerturbedT> X);

// Given polynomial numerator and denominator functions, evaluate numerator(X+epsilon)/denominator(X+epsilon) rounded
// to int for the same infinitesimal perturbation epsilon as in perturbed_sign.  The numerator and denominator must be
// multivariate polynomials of at most the given degree.  The rational function is packed as
//
//   ratio(X) = concatenate(numerator(X),denominator(X)).
//
// Rounding is to nearest.
//
// If the rounded result does not fit into the range [-bound,bound] (e.g., if its infinite), an exception is thrown.
// This will never happen for appropriately shielded predicates, such as constructing the intersection of two segments
// once perturbed_sign has verified that they intersect after perturbation.
//
// If take_sqrt is true, an exactly rounded square root is computed.
// The r+1 numbers (r numerators and one denominator) should be copied into the result array via r+1 calls to mpz_set.
// The perturbed sign of the denominator (before any square root) is returned.
template<class PerturbedT> GEODE_CORE_EXPORT GEODE_COLD bool
perturbed_ratio(RawArray<Quantized> result,
                void(*const ratio)(RawArray<mp_limb_t,2>,RawArray<const Vector<Exact<1>,PerturbedT::m>>),
                const int degree, RawArray<const PerturbedT> X, const bool take_sqrt=false);

// The levelth perturbation of point i in R^m.  This is exposed for occasional special purpose use only, or as a
// convenient pseudorandom generator; normally this routine is called internally by perturbed_sign.  perturbation<m+1>
// starts with perturbation<m>.
template<int m> GEODE_CORE_EXPORT Vector<ExactInt,m> perturbation(const int level, const int i);

// The type of a degree d predicate evaluated on vectors of the given type (see usage in predicates.cpp)
template<int d,class TV> struct PredicateTypeHelper { typedef typename TV::Scalar type; };
template<int d,int m> struct PredicateTypeHelper<d,Vector<Exact<1>,m>> { typedef Exact<d> type; };
template<int d,class TV> using PredicateType = typename PredicateTypeHelper<d,TV>::type;
template<int n,int d,class TV> using ConstructType = Tuple<Vector<PredicateType<d,TV>,n>,PredicateType<d-1,TV>>;

// Wrap predicate for consumption by perturbed_sign (use only via perturbed_predicate)
template<class R> struct WrapResult { typedef RawArray<mp_limb_t> type; };
template<class... Rs> struct WrapResult<Tuple<Rs...>> { typedef RawArray<mp_limb_t,2> type; };
template<class F,int d,class... entries> static void
wrapped_predicate(typename WrapResult<decltype(F::eval(
                    declval<typename First<const Vector<Exact<1>,d>,entries>::type>()...))>::type result,
                  RawArray<const Vector<Exact<1>,d>> X) {
  assert(X.size()==sizeof...(entries));
  mpz_set(result,F::eval(X[entries::value]...));
}
template<class F,int d,class... entries> GEODE_ALWAYS_INLINE static inline auto
wrap_predicate(Types<entries...>) -> decltype(&wrapped_predicate<F,d,entries...>) {
  return &wrapped_predicate<F,d,entries...>;
}

// Given F s.t. F::eval exactly computes a polynomial in its input arguments, compute the perturbed sign of F(args).
// This is the standard way of turning an expression into a perturbed predicate.  For examples, see predicates.cpp.
template<class F,class... Args> GEODE_ALWAYS_INLINE static inline bool perturbed_predicate(const Args... args) {
  typedef typename First<Args...>::type PerturbedT;
  const int d = PerturbedT::m;
  typedef decltype(F::eval(Vector<Exact<1>,d>(args.value())...)) Result;
  const int degree = Result::degree;

  // Check irreducibility if desired
  const auto f = wrap_predicate<F,d>(IRange<sizeof...(Args)>());
  if (IRREDUCIBLE)
    inexact_assert_irreducible(f,degree,sizeof...(Args),typeid(F).name());

  // Evaluate with conservative interval arithmetic, hoping for a clear nonzero
  if (const int s = weak_sign(F::eval(Vector<Interval,d>(args.value())...)))
    return s>0;

  // Fall back to exact integer evaluation with symbolic perturbation
  const PerturbedT X[sizeof...(Args)] = {args...};
  return perturbed_sign(f,degree,asarray(X));
}

template<class F,class... Args> struct PerturbedConstruct {
  static const int d = First<Args...>::type::m;
  typedef decltype(F::eval(Vector<Exact<1>,d>(declval<Args>().value())...)) Result;
  static const int degree = Result::second_type::degree+1;
  static_assert(Result::first_type::value_type::degree==degree,
                "We require degree d/(d-1) so that the result has degree 1.");
  static const int k = Result::first_type::m;
  typedef Tuple<Vector<Quantized,k>,bool> type;
};

// Given F s.t. F::eval computes tuple(x,y) representing x/y, perform the perturbed construction of x/y accurate
// to within the given tolerance.  Returns (x/y,sign(y)).
template<class F,class... Args> GEODE_ALWAYS_INLINE static inline typename PerturbedConstruct<F,Args...>::type
perturbed_construct(const int tolerance, const Args... args) {
  typedef PerturbedConstruct<F,Args...> I;

  // Check irreducibility if desired
  const auto f = wrap_predicate<F,I::d>(IRange<sizeof...(Args)>());
  if (IRREDUCIBLE)
    inexact_assert_lowest_terms(f,I::degree,sizeof...(Args),I::d,typeid(F).name());

#if CHECK
  Tuple<Vector<Interval,I::k>,int> check;
  check.x.fill(Interval::full());
#endif

  // Evaluate with intervals first
  {
    const auto i = F::eval(Vector<Interval,I::d>(args.value())...);
    if (!i.y.contains_zero()) {
      const auto r = inverse(i.y)*i.x;
      const int s = certainly_positive(i.y) ? 1 : -1;
#if CHECK
      check = tuple(r,s);
#else
      if (small(r,tolerance))
        return tuple(snap(r),s>0);
#endif
    }
  }

  // If intervals fail, evaluate and round using symbolic perturbation
  const typename First<Args...>::type X[sizeof...(Args)] = {args...};
  Vector<Quantized,I::k> q;
  const bool s = perturbed_ratio(asarray(q),f,I::degree,asarray(X));
#if CHECK
  GEODE_ASSERT(!check.y || (check.y>0)==s);
  for (int i=0;i<I::k;i++)
    GEODE_ASSERT(check.x[i].thickened(.5).contains(q[i]));
#endif
  return tuple(q,s);
}

// Compute the sign of A + sign*B*sqrt(C), where A,B,C are polynomials and |sign| = 1.  C must be positive.
namespace {
template<class A,class B,class C> struct OneSqrt {
  template<class... Args> static auto eval(const Args... args) // A^2 - B^2 C
    -> decltype(sqr(A::eval(args...))-sqr(B::eval(args...))*C::eval(args...)) {
    return sqr(A::eval(args...))-sqr(B::eval(args...))*C::eval(args...);
  }
};}
template<class A,class B,class C,class... Args> GEODE_ALWAYS_INLINE static inline bool
perturbed_predicate_sqrt(const int sign, const Args... args) {
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

// Compute the sign of A + B0*sign0*sqrt(C0) + B1*sign1*sqrt(C1), where A,B0,B1,C0,C1 are polynomials and
// |sign0| = |sign1| = 1.  C0 and C1 must be positive.
namespace {
template<class A,class B0,class B1,class C0,class C1> struct TwoSqrtsAlpha {
  template<class... Args> static auto eval(const Args... args)
    -> decltype(sqr(A::eval(args...))) {
    return sqr(A::eval(args...)) + sqr(B0::eval(args...))*C0::eval(args...)
                                 - sqr(B1::eval(args...))*C1::eval(args...);
  }
};
template<class A,class B0> struct TwoSqrtsBeta { template<class... Args> static auto eval(const Args... args)
  -> decltype(A::eval(args...)*(B0::eval(args...)<<1)) {
  return A::eval(args...)*(B0::eval(args...)<<1);
}};}
template<class A,class B0,class B1,class C0,class C1,class... Args> GEODE_ALWAYS_INLINE static inline bool
perturbed_predicate_two_sqrts(const int sign0, const int sign1, const Args... args) {
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
  // Otherwise, evaluate an even larger predicate.  For completeness, this larger predicate has the form
  // (ignoring sign flips)
  //   G + sign1*B1*sqrt(C1) > 0   iff   A^2 + B0^2*C0 - B1^2*C1 > -2*A*B0*sign0*sqrt(C0)
  //                               iff   (A^2 + B0^2*C0 - B1^2*C1)^2 > 4*A^2*B0^2*C0
  //                               iff   A^4 + B0^4*C0^2 + B1^4*C1^2 + 2*A^2*B0^2*C0 - 2*A^2*B1^2*C1 - 2*B0^2*B1^2*C0*C1 > 4*A^2*B0^2*C0
  //                               iff   A^4 + B0^4 C0^2 + B1^4 C1^2 - 2 A^2 B0^2 C0 - 2 A^2 B1^2 C1 - 2 B0^2 B1^2 C0 C1 > 0
  return perturbed_predicate<OneSqrt<Alpha,Beta,C0>>(args...) ^ (sAlpha < 0) ^ (sG < 0);
}

void snap_divs(RawArray<Quantized> result, RawArray<mp_limb_t,2> values, const bool take_sqrt);

template<int a, int b> Quantized snap_div(const Exact<a> num, const Exact<b> denum, const bool take_sqrt) {
  enum { max_size = (a<b ? b : a), n_terms = 2 };
  // Pack both values into a contigous array
  auto packed_values = Vector<Exact<max_size>, n_terms>(Exact<max_size>(num), Exact<max_size>(denum));
  static_assert(sizeof(packed_values) == sizeof(mp_limb_t)*max_size*n_terms,
                "Memory layout doesn't appear to be correct");
  Vector<Quantized,n_terms-1> result;
  snap_divs(asarray(result),RawArray<mp_limb_t,2>(n_terms,max_size,(mp_limb_t*)packed_values.data()),take_sqrt);
  return result.x;
}

// Compute exactly rounded rational function
// denom must not be zero
template<int a, int b> Vector<Quantized,2> snap_div(const Vector<Exact<a>,2> num, const Exact<b> denum,
                                                    const bool take_sqrt) {
  enum { max_size = (a<b ? b : a), n_terms = 3 };
  // Pack both values into a contigous array
  Vector<Exact<max_size>,n_terms> packed_values(Exact<max_size>(num.x),Exact<max_size>(num.y),Exact<max_size>(denum));
  static_assert(sizeof(packed_values) == sizeof(mp_limb_t)*max_size*n_terms,
                "Memory layout doesn't appear to be correct");
  Vector<Quantized,n_terms-1> result;
  snap_divs(asarray(result),RawArray<mp_limb_t,2>(n_terms,max_size,(mp_limb_t*)packed_values.data()),take_sqrt);
  return result;
}

// For testing purposes
GEODE_CORE_EXPORT Array<const uint8_t,2> perturb_monomials(const int degree, const int variables);
template<int m> GEODE_CORE_EXPORT void perturbed_sign_test();
GEODE_CORE_EXPORT void in_place_interpolating_polynomial_test(const int degree, RawArray<const uint8_t,2> lambda,
                                                              RawArray<const ExactInt> coefs, const bool verbose);
GEODE_CORE_EXPORT Array<Quantized> snap_divs_test(RawArray<mp_limb_t,2> values, const bool take_sqrt);
GEODE_CORE_EXPORT void perturbed_ratio_test();

}
