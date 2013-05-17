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
#include <other/core/structure/Tuple.h>
#include <other/core/vector/Vector.h>
#include <stdint.h>
namespace other {
namespace exact {

// Integer values in [-bound,bound] are safely exactly representable.  To allow a bit of
// slack for algorithms to use, all quantized points will live in roughly [-bound,bound]/1.01.
const int log_bound = 24;
const int bound = (1<<log_bound)-1;

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

// Like CGAL, GMP assumes that the C++11 standard library exists whenever C++11 does.  This is false for clang.
#define __GMPXX_USE_CXX11 0

// Typedefs for indexed points
template<int d> struct Point {
  typedef Tuple<int,Vector<exact::Int,d>> type;
  BOOST_STATIC_ASSERT(sizeof(type)==sizeof(int)+d*sizeof(exact::Int));
};
typedef Vector<exact::Int,2> Vec2;
typedef Vector<exact::Int,3> Vec3;
typedef typename Point<2>::type Point2;
typedef typename Point<3>::type Point3;

}
}
