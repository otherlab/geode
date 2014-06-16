//#####################################################################
// Class RawArray<T,d>
//#####################################################################
#include <geode/array/RawArray.h>
#include <geode/array/Array2d.h>
#include <geode/python/numpy.h>
#include <geode/vector/Vector.h>
#include <geode/utility/process.h>
#if defined(GEODE_BLAS) && defined(GEODE_MKL)
#include <geode/vector/blas.h>
#include <mkl_trans.h>
#endif
namespace geode {
using std::cerr;
using std::endl;

template<class T> void RawArray<T,2>::
transpose() {
  if (m==n)
    for (int i=0;i<m;i++) for (int j=0;j<i;j++)
      swap(flat[i*n+j],flat[j*n+i]);
  else {
    Array<T,2> copy(transposed());
    swap(n,m);
    *this = copy;
  }
}

#if defined(GEODE_BLAS) && defined(GEODE_MKL)
template<> void RawArray<float,2>::transpose() {
  imatcopy(1,*this);
  swap(n,m);
}
template<> void RawArray<double,2>::transpose() {
  imatcopy(1,*this);
  swap(n,m);
}
#endif

template<class T> Array<typename remove_const<T>::type,2> RawArray<T,2>::transposed() const {
  Array<Element,2> A(n,m,uninit);
  for (int i=0;i<m;i++) for (int j=0;j<n;j++)
    A(j,i) = flat[i*n+j];
  return A;
}

#if defined(GEODE_BLAS) && defined(GEODE_MKL)
template<> Array<float,2> RawArray<float,2>::transposed() const {
  Array<Element,2> A(n,m,uninit);
  omatcopy(CblasTrans,1,*this,A);
  return A;
}
template<> Array<float,2> RawArray<const float,2>::transposed() const {
  Array<Element,2> A(n,m,uninit);
  omatcopy(CblasTrans,1,*this,A);
  return A;
}
template<> Array<double,2> RawArray<double,2>::transposed() const {
  Array<Element,2> A(n,m,uninit);
  omatcopy(CblasTrans,1,*this,A);
  return A;
}
template<> Array<double,2> RawArray<const double,2>::transposed() const {
  Array<Element,2> A(n,m,uninit);
  omatcopy(CblasTrans,1,*this,A);
  return A;
}
#endif

template<class T> void RawArray<T,2>::permute_rows(RawArray<const int> p, int direction) const {
  // Not an exhaustive assertion, but should be sufficient:
  assert(abs(direction)==1 && p.size()==m && p.sum()==m*(m-1)/2);
  for (int i=0;i<m;i++) assert(unsigned(p[i])<unsigned(m));

  // Convert to forward case if necessary
  Array<int> q(m,uninit);
  if (direction<0) // backwards
    for (int i=0;i<m;i++)
      q[p[i]] = i;
  else
    q.copy(p);

  // Now we can assume the forward case
  Array<T> tmp(n,uninit);
  T* data_ = data();
  for (int i=0;i<m;i++)
    if (q[i]>=0 && q[i]!=i) {
      memcpy(tmp.data(),data_+i*n,n*sizeof(T));
      int e = i;
      do {
        memcpy(data_+e*n,data_+q[e]*n,n*sizeof(T));
        int qe = q[e];
        q[e] = -1;
        e = qe;
      } while (q[e]!=i);
      memcpy(data_+e*n,tmp.data(),n*sizeof(T));
      q[e] = -1;
    }
}

template<class T> void RawArray<T,2>::permute_columns(RawArray<const int> p,int direction) const {
  // Not an exhaustive assertion, but should be sufficient:
  assert(abs(direction)==1 && p.size()==n && p.sum()==n*(n-1)/2);
  for (int i=0;i<n;i++) assert(unsigned(p[i])<unsigned(n));

  Array<T> tmp(n,uninit);
  for (int i=0;i<m;i++) {
    memcpy(tmp.data(),data()+i*n,n*sizeof(T));
    if (direction>0) // forwards
      for (int j=0;j<n;j++)
        flat[i*n+j] = tmp[p[j]];
    else // backwards
      for (int j=0;j<n;j++)
        flat[i*n+p[j]] = tmp[j];
  }
}

real frobenius_norm(RawArray<const real,2> A) {
#if defined(GEODE_BLAS) && defined(GEODE_MKL)
  return nrm2(A.flat);
#else
  return A.flat.sqr_magnitude();
#endif
}

real infinity_norm(RawArray<const real,2> A) {
  real norm = 0;
  for (int i=0;i<A.m;i++)
    norm = max(norm,A[i].maxabs());
  return norm;
}

#if !(defined(GEODE_BLAS) && defined(GEODE_MKL) && defined(_WIN32))
template void RawArray<float,2>::transpose();
template void RawArray<double,2>::transpose();
template Array<float,2> RawArray<float,2>::transposed() const;
template Array<double,2> RawArray<double,2>::transposed() const;
template Array<float,2> RawArray<const float,2>::transposed() const;
template Array<double,2> RawArray<const double,2>::transposed() const;
#endif
template void RawArray<unsigned char,2>::transpose();
template void RawArray<Vector<float,3>,2>::transpose();
template void RawArray<Vector<float,4>,2>::transpose();
template void RawArray<Vector<double,3>,2>::transpose();
template void RawArray<Vector<double,4>,2>::transpose();
template void RawArray<Vector<unsigned char,3>,2>::transpose();
template void RawArray<Vector<unsigned char,4>,2>::transpose();
template Array<int,2> RawArray<const int,2>::transposed() const;
template Array<Vector<unsigned char,3>,2> RawArray<Vector<unsigned char,3>,2>::transposed() const;
template Array<Vector<unsigned char,4>,2> RawArray<Vector<unsigned char,4>,2>::transposed() const;
template void RawArray<real,2>::permute_rows(RawArray<const int>,int) const;
template void RawArray<real,2>::permute_columns(RawArray<const int>,int) const;

TemporaryOwner::TemporaryOwner()
  : owner(new_<Object>()) {
  GEODE_ASSERT(owner->ob_refcnt==1);
}

TemporaryOwner::~TemporaryOwner() {
  if (owner->ob_refcnt!=1) {
    cerr<<"fatal: Reference to temporary memory area stolen by python (ob_refcnt = "
        <<owner->ob_refcnt<<" != 1)."<<endl;
    process::backtrace();
    GEODE_FATAL_ERROR("Reference to temporary memory area stolen by python");
  }
}

}
