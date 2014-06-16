#pragma once

#include <geode/array/Array.h>

namespace geode {

// applies a stencil function f(a,(x,...)) to an array a in-place. a(x,...) will be
// assigned f(a,(x,...)). f(a,(x...)) is guaranteed to only access elements of a
// within the x range [x-w, oo); the stencil can only "look back" by at most w.
// The intermediate results (values for f(a,(x,...)) for coordinates that may still
// be used by other calls to f) are stored in a ring-buffer of dimensions (w,s1,...)
// where a has dimensions (s0,s1,...).
// Because array storage is such that the first coordinate has the largest offsets,
// both a and the ring-buffer can be addressed using the same offsets, which a
// stencil function f can precompute.
// TODO: write a small adaptor such that F can use a flat index into a directly,
// assuming it has already computed offsets and can deal with index conversion
// alone.
template<class F, class T, int d>
void apply_stencil(F &f, int w, const Array<T,d> a) {
  // make working memory
  auto bdims = a.sizes();
  bdims[0] = w+1;
  Array<T,d> b(bdims,uninit);
  // apply the stencil
  apply_stencil(f, w, a, b);
}

// as above, b is ring buffer for storing results while the original values
// are still needed for other evaluations
template<class F, class T, int d>
void apply_stencil(F &f, int w, const Array<T,d> a, const Array<T,d> b) {
  // make sure working memory is of the right size
  auto adims = b.sizes();
  GEODE_ASSERT(adims[0] == w+1);
  adims[0] = a.sizes()[0];
  GEODE_ASSERT(adims == a.sizes());

  // traverse a along first dimension
  int s = adims[0];
  for (int i = 0; i < s+(w+1); ++i) {
    // slice i is bi in b
    int bi = i%(w+1);

    // write back the row we are going to overwrite to where it belongs
    int aj = i-(w+1);

    // make a proxy array so we get the index from a flat index
    auto slice = a[0];

    // traverse along other directions, and store in b
    int n = a.sizes().template slice<1,d>().product();
    for (int k = 0; k < n; ++k) {
      auto idx = slice.index(k);
      auto bidx = idx.insert(bi, 0);

      // write back if we're far enough out
      if (aj >= 0) {
        auto ajdx = idx.insert(aj, 0);
        a[ajdx] = b[bidx];
      }

      // compute new if still new slices left
      if (i < s) {
        auto aidx = idx.insert(i, 0);
        b[bidx] = f(a, aidx);
      }
    }
  }
}

// This is a sample stencil that computes the maximum over a spherical area of
// radius r (on a 2D array)
template<class T>
class MaxStencil: public Object {
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef T value_type;

  int r;
  MaxStencil(int r): r(r) {};

  T operator()(const Array<const T,2> a, Vector<int,2> const &idx) {
    T v = a[idx];

    // this is pretty dumb, but hey.
    for (int i = -r; i < r; ++i) {
      for (int j = -r; j < r; ++j) {
        if (i*i + j*j <= r*r) {
          Vector<int,2> ii(idx.x+i, idx.y+j);
          if (a.valid(ii))
            v = max(v, a[ii]);
        }
      }
    }

    return v;
  }
};

}
