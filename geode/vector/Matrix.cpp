//#####################################################################
// Class Matrix
//#####################################################################
#include <geode/vector/Matrix.h>
#include <geode/vector/DiagonalMatrix.h>
#include <geode/array/view.h>
#include <geode/array/NdArray.h>
#include <geode/utility/str.h>
namespace geode {

NdArray<real> fast_singular_values(const NdArray<const real>& A) {
  GEODE_ASSERT(A.rank()>=2);
  const int r = A.rank();
  if (!(A.shape[r-2]==A.shape[r-1] && 2<=A.shape[r-1] && A.shape[r-1]<=3))
    throw NotImplementedError(format(
      "fast_singular_values: got shape %s, only arrays of 2x2 and 3x3 matrices implemented for now",str(A.shape)));
  NdArray<real> D(concatenate(A.shape.slice(0,r-2),asarray(vec(min(A.shape[r-1],A.shape[r-2])))),uninit);
  const auto Ds = D.flat.reshape(D.flat.size()/D.shape.back(),D.shape.back());
  switch (A.shape[r-1]) {
    #define CASE(d) \
      case d: { \
        const auto As = vector_view<Matrix<real,d>>(A.flat); \
        for (const int i : range(Ds.m)) \
          Ds[i] = asarray(fast_singular_values(As[i]).to_vector()); \
        break; \
      }
    CASE(2) CASE(3)
    #undef CASE
    default: GEODE_UNREACHABLE();
  }
  return D;
}

}
