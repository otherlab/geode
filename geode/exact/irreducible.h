// Irreducibility testing for integer polynomials
#pragma

// The efficiency of the symbolic perturbation core depends cubically on the degrees
// of polynomial predicates, so it is useful to know whether said degrees can be reduced.
// This code implements a randomized algorithm for checking irreducibility over Q.
// To turn on checking for all predicates, flip the IRREDUCIBILITY flag.

// Turn on to check whether all predicates and constructions are irreducible.
#define IRREDUCIBLE 0

#include <geode/exact/Exact.h>
namespace geode {

// Check if a polynomial is irreducible, and bail if not.  Reducible polynomials are
// always correctly recognized as such, but some irreducible polynomials will be treated
// as irreducible.  Use with care.
template<int m> void
inexact_assert_irreducible(void(*const polynomial)(RawArray<mp_limb_t>,RawArray<const Vector<Exact<1>,m>>),
                           const int degree, const int inputs, const char* name);

// Check if a degree 1 rational function is in lowest terms
template<int m> void
inexact_assert_lowest_terms(void(*const ratio)(RawArray<mp_limb_t,2>,RawArray<const Vector<Exact<1>,m>>),
                            const int degree, const int inputs, const int outputs, const char* name);

// For testing purposes
void irreducible_test();

}
