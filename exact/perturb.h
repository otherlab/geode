// General purpose black box simulation of simplicity
#pragma once

#include <other/core/exact/config.h>
#include <other/core/structure/Tuple.h>
#include <other/core/vector/Vector.h>
#include <gmpxx.h>
namespace other {

// Assuming predicate(X) == 0, evaluate predicate(X+epsilon)>0 for a certain infinitesimal perturbation epsilon.  The predicate
// must be a multivariate polynomial of at most the given degree.  The permutation chosen is deterministic, independent of the
// predicate, and guaranteed to work for all possible polynomials.  Each coordinate of the X array contains (1) the value and
// (2) the index of the value, which is used to look up the fixed perturbation.
template<int m> OTHER_CORE_EXPORT OTHER_COLD bool perturbed_sign(mpz_class (*const predicate)(RawArray<const Vector<exact::Int,m>>), const int degree, RawArray<const Tuple<int,Vector<exact::Int,m>>> X);

}
