//#####################################################################
// Class Frame
//#####################################################################
#include <geode/vector/Frame.h>
namespace geode {

typedef real T;

template<class TV> Frame<TV> frame_test(const Frame<TV>& f1, const Frame<TV>& f2, const TV x) {
  return f1*f2*Frame<TV>(x);
}

template<class TV> Array<Frame<TV>> frame_array_test(const Frame<TV>& f1, Array<const Frame<TV>> f2, const TV x) {
    Array<Frame<TV>> ff(f2.size());
    for (int i=0;i<f2.size();i++)
        ff[i] = f1*f2[i]*Frame<TV>(x);
    return ff;
}

template<class TV> Array<Frame<TV>> frame_interpolation(Array<const Frame<TV>> f1, Array<const Frame<TV>> f2, const T s) {
  GEODE_ASSERT(f1.size()==f2.size());
  Array<Frame<TV>> r(f1.size(),uninit);
  for (int i=0;i<r.size();i++)
    r[i] = Frame<TV>::interpolation(f1[i],f2[i],s);
  return r;
}

}
