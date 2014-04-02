//#####################################################################
// Class Rotation
//#####################################################################
#include <geode/vector/Matrix.h>
#include <geode/vector/Rotation.h>
#include <geode/array/NdArray.h>
namespace geode {

typedef real T;

template<class TV> Rotation<TV> rotation_test(const Rotation<TV>& r) {
    return r*r;
}

template<class TV> Array<Rotation<TV>> rotation_array_test(Array<const Rotation<TV>> r) {
    Array<Rotation<TV>> rr(r.size());
    for (int i=0;i<r.size();i++)
        rr[i] = sqr(r[i]);
    return rr;
}

#if 0
PyObject* rotation_from_matrix(NdArray<const real> A) {
  GEODE_ASSERT(A.rank()>=2);
  const int r = A.rank();
  if (A.shape[r-1]==2 && A.shape[r-2]==2) {
    NdArray<Rotation<Vector<T,2>>> rs(A.shape.slice_own(0,r-2),uninit);
    for (const int i : range(rs.flat.size()))
      rs.flat[i] = Rotation<Vector<T,2>>(Matrix<real,2>(A.flat.slice(4*i,4*(i+1)).reshape(2,2)));
    return to_python(rs);
  } else if (A.shape[r-1]==3 && A.shape[r-2]==3) {
    NdArray<Rotation<Vector<T,3>>> rs(A.shape.slice_own(0,r-2),uninit);
    for (const int i : range(rs.flat.size()))
      rs.flat[i] = Rotation<Vector<T,3>>(Matrix<real,3>(A.flat.slice(9*i,9*(i+1)).reshape(3,3)));
    return to_python(rs);
  } else
    throw TypeError(format("expected 2x2 or 3x3 matrices, got shape %s",str(A.shape)));
}
#endif

NdArray<Rotation<Vector<T,3>>> rotation_from_euler_angles_3d(NdArray<const Vector<T,3>> theta) {
  NdArray<Rotation<Vector<T,3>>> R(theta.shape,uninit);
  for (const int i : range(R.flat.size()))
    R.flat[i] = Rotation<Vector<T,3>>::from_euler_angles(theta.flat[i]);
  return R;
}

NdArray<Vector<T,3>> rotation_euler_angles_3d(NdArray<const Rotation<Vector<T,3>>> R) {
  NdArray<Vector<T,3>> theta(R.shape,uninit);
  for (const int i : range(R.flat.size()))
    theta.flat[i] = R.flat[i].euler_angles();
  return theta;
}

}
