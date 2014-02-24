// Array permutation routines
#pragma once

#include <geode/array/Array.h>
namespace geode {

// dst[perm[i]] = src[i]
template<class D,class S,class P> void permute(D& dst, const S& src, const P& perm) {
  STATIC_ASSERT_SAME(typename D::value_type,typename S::value_type);
  const int m = perm.size();
  GEODE_ASSERT(dst.size()==m && src.size()==m);
  for (int i=0;i<m;i++)
    dst[perm[i]] = src[i];
}

// dst[i] = src[perm[i]]
template<class D,class S,class P> void unpermute(D& dst, const S& src, const P& perm) {
  STATIC_ASSERT_SAME(typename D::value_type,typename S::value_type);
  const int m = perm.size();
  GEODE_ASSERT(dst.size()==m && src.size()==m);
  for (int i=0;i<m;i++)
    dst[i] = src[perm[i]];
}

// dst[perm[i]] = src[i], skipping perm[i]<0 entries and compacting dst
template<class D,class S,class P> void partial_permute(D& dst, const S& src, const P& perm) {
  STATIC_ASSERT_SAME(typename D::value_type,typename S::value_type);
  const int m = perm.size();
  GEODE_ASSERT(src.size()==m);
  dst.resize(m ? perm.max()+1 : 0);
  for (int i=0;i<m;i++) {
    const int pi = perm[i]; 
    if (pi >= 0)
      dst[pi] = src[i];
  }
}

template<class A,class P,class W> void inplace_partial_permute(A& x, const P& perm, W& work) {
  partial_permute(work,x,perm);
  x.copy(work);
}

// Requires x.size()==block*perm.size()
GEODE_EXPORT void inplace_partial_permute(UntypedArray& x, RawArray<const int> perm,
                                          Array<char>& work, const int block=1);
template<class P> static inline void inplace_partial_permute(UntypedArray& x, const P& perm,
                                                             Array<char>& work, const int block=1) {
  inplace_partial_permute(x,RawArray<const int>(perm),work,block);
}

}
