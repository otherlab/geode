// Warp and quantize a set of points in R^d in preparation for exact arithmetic
#pragma once

/* Doubles have 52 bits of mantissa, so they can exactly represent any integer in [-2**53,2**53]
 * (52+1 due to the implicit 1 before the mantissa).  Single precision floats can handle
 * [-2**24,2**24].  To give this some context, 25 bit precision in a 10 meter space is an accuracy 
 * of 0.3 um.  This should be sufficient for our purposes for now.
 *
 * Thus, in order to perform exact arithmetic on points within a given box, we first warp the
 * box into the range [1-2**24,2**24-1], then quantize to int32_t, then convert back to float.
 * The result is a number than be operated on with fast conservative interval arithmetic and
 * also exactly converted to int32_t for exact integer math.
 */

#include <other/core/geometry/Box.h>
#include <limits>
namespace other {

template<class TS,int d> struct Quantizer {
  typedef float T;
  typedef int32_t Int;
  typedef Vector<T,d> TV;  // quantized vector type
  typedef Vector<TS,d> TVS; // unquantized vector type

  struct Inverse {
    const TVS center;
    const TS inv_scale; 

    Inverse(TVS center, TS inv_scale)
      : center(center), inv_scale(inv_scale) {}

    TVS operator()(const TV& p) const {
      return center+(inv_scale*TVS(p));
    }
  };

  const TVS center;
  const TS scale;
  const Inverse inverse;

  Quantizer(const Box<TVS>& box)
    : center(box.center())
    , scale(ldexp((TS)1-8*numeric_limits<T>::epsilon(),numeric_limits<T>::digits+1)/box.sizes().max()) // 8 is a bit slack, but no one cares
    , inverse(center,1/scale) {}

  TV operator()(const TVS& p) const {
    TV q(scale*(p-center)); // Transform to 1-2**24 <= q <= 2**24-1
    for (int i=0;i<d;i++)
      q[i] = Int(q[i]);
    return q;
  }
};

template<class TS,int d> static inline Quantizer<TS, d> quantizer(const Box<Vector<TS,d>>& box) {
  return Quantizer<TS,d>(box);
}

}
