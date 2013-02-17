// Warp and quantize a set of points in R^d in preparation for exact arithmetic
// See other/core/exact/config.h for discussion.
#pragma once

#include <other/core/exact/config.h>
#include <other/core/geometry/Box.h>
#include <limits>
namespace other {

template<class TS,int d> struct Quantizer {
  typedef exact::Real T;
  typedef exact::Int Int;
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
    , scale(ldexp((TS)1-8*numeric_limits<T>::epsilon(),numeric_limits<T>::digits+1)/max(1.01*box.sizes().max(),T(1e-6))) // 8 is a bit slack, but no one cares
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
