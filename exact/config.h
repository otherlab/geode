// Multiprecision integer arithmetic for exact geometric predicates
#pragma once 

/* Doubles have 52 bits of mantissa, so they can exactly represent any integer in [-2**53,2**53]
 * (52+1 due to the implicit 1 before the mantissa).  Single precision floats can handle
 * [-2**24,2**24].  To give this some context, 25 bit precision in a 10 meter space is an accuracy 
 * of 0.3 um.  This should be sufficient for our purposes for now.
 *
 * Thus, in order to perform exact arithmetic on points within a given box, we first warp the
 * box into the range [1-2**24,2**24-1]/1.01, then quantize to int32_t, then convert back to float.
 * The result is a number that can be operated on with fast conservative interval arithmetic and
 * also exactly converted to int32_t for exact integer math.  The 1.01 factor gives algorithms a
 * bit of space for sentinel purposes.
 *
 * Actually, Delaunay runs 20% faster without the interval steps, so we'll go pure integer for now.
 */

#include <other/core/utility/config.h>
#include <stdint.h>
namespace other {
namespace exact {

// Integer values in [-bound,bound] are safely exactly representable.  To allow a bit of
// slack for algorithms to use, all quantized points will live in roughly [-bound,bound]/1.01.
const int bound = (1<<24)-1;

// Base integer type for exact arithmetic
typedef int32_t Int;

// Use pure integer math for now
#define OTHER_EXACT_INTERVAL_FILTER 0

#if OTHER_EXACT_INTERVAL_FILTER
// Floating point type for exact arithmetic.  All values used for exact arithmetic will be exact integers.
typedef float Real;
typedef Real Quantized;
#else
typedef Int Quantized;
#endif

}
}
