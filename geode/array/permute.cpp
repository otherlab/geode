// Array permutation routines

#include <geode/array/permute.h>
#include <geode/array/UntypedArray.h>
namespace geode {

void inplace_partial_permute(UntypedArray& x, RawArray<const int> perm, Array<char>& work, const int block) {
  const int m = perm.size();
  const int b_size = block*x.t_size();
  GEODE_ASSERT(x.size()==block*m);
  const int space = m ? (perm.max()+1)*b_size : 0;
  work.resize(space);
  for (int i=0;i<m;i++) {
    const int pi = perm[i];
    if (pi >= 0)
      memcpy(work.data()+pi*b_size,x.data()+i*b_size,b_size);
  }
  x.resize(space/x.t_size());
  memcpy(x.data(),work.data(),space);
}

}
