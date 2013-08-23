// Multiprecision integer arithmetic for exact geometric predicates
#pragma once

/* Doubles have 52 bits of mantissa, so they can exactly represent any integer in [-2**53,2**53]
 * (52+1 due to the implicit 1 before the mantissa).  Single precision floats can handle
 * [-2**24,2**24].  To give this some context, 25 bit precision in a 10 meter space is an accuracy
 * of 0.3 um.  Ideally, this would be sufficient for our purposes for now.
 *
 * Unfortunately, some algorithms require additional sentinel bits for special purposes, so 2**24
 * is a bit too tight.  Symbolic perturbation of high degree predicates absorbs a few more bits.
 * While its often possible to carefully arrange for single precision to work, double precision is
 * easy and more precise anyways.  Thus, we quantize into the integer range [-2**53,2**53]/1.01.
 * The 1.01 factor gives algorithms a bit of space for sentinel purposes.
 */

#include <other/core/utility/config.h>
#include <other/core/structure/Tuple.h>
#include <other/core/vector/Vector.h>
#include <stdint.h>
namespace other {
namespace exact {

// Integer values in [-bound,bound] are safely exactly representable.  To allow a bit of
// slack for algorithms to use, all quantized points will live in roughly [-bound,bound]/1.01.
const int log_bound = 53;
const int64_t bound = (int64_t(1)<<log_bound)-1;

}

// Base integer type for exact arithmetic
#define OTHER_EXACT_INT 64
typedef int64_t ExactInt;
typedef double Quantized;

namespace exact {

// Like CGAL, GMP assumes that the C++11 standard library exists whenever C++11 does.  This is false for clang.
#define __GMPXX_USE_CXX11 0

// Typedefs for indexed points
template<int d> struct Point {
  typedef Tuple<int,Vector<Quantized,d>> type;
  BOOST_STATIC_ASSERT(sizeof(type)==sizeof(int)+4+d*sizeof(Quantized));
};
typedef Vector<Quantized,2> Vec2;
typedef Vector<Quantized,3> Vec3;
typedef typename Point<2>::type Point2;
typedef typename Point<3>::type Point3;

}
}
