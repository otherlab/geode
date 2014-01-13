//#####################################################################
// Class Array<T,3>
//#####################################################################
//
// A shareable, three-dimensional array.
//
//#####################################################################
#pragma once

#include <geode/array/ArrayNdBase.h>
#include <geode/array/Subarray.h>
#include <geode/geometry/Box.h>
#include <geode/python/exceptions.h>
#include <geode/vector/Vector.h>
namespace geode {

template<class T_>
class Array<T_,3> : public ArrayNdBase<T_,Array<T_,3> >
{
  typedef T_ T;
public:
  enum Workaround1 {dimension=3};
  enum Workaround2 {d=dimension};
  typedef typename remove_const<T>::type Element;
  typedef ArrayNdBase<T,Array> Base;

  using Base::flat; // one-dimensional data storage
  using Base::data;

private:
  struct Unusable{};
  typedef typename mpl::if_<is_const<T>,Array<Element,d>,Unusable>::type MutableSelf;
public:

  int m, n, mn; // sizes

  Array()
    : m(0), n(0), mn(0) {}

  Array(const Vector<int,d>& counts, const bool initialize=true)
    : Base(counts.x*counts.y*counts.z,initialize), m(counts.x), n(counts.y), mn(counts.z) {
    assert(m>=0 && n>=0 && mn>=0);
  }

  Array(const int m, const int n, const int mn, const bool initialize=true)
    : Base(m*n*mn,initialize),m(m),n(n),mn(mn) {
    assert(m>=0 && n>=0 && mn>=0);
  }

  Array(const int m, const int n, const int mn, T* data, PyObject* owner)
    : m(m), n(n), mn(mn) {
    assert(m>=0 && n>=0 && mn>=0);
    flat = Array<T>(m*n*mn,data,owner);
  }

  Array(const Vector<int,d>& counts, T* data, PyObject* owner)
    : m(counts.x), n(counts.y), mn(counts.z) {
    assert(m>=0 && n>=0 && mn>=0);
    flat = Array<T>(m*n*mn,data,owner);
  }

  Array(const Array& source)
    : m(source.m), n(source.n), mn(source.mn) {
    flat = source.flat;
  }

  // Conversion from mutable to const
  Array(const MutableSelf& source)
    : m(source.m), n(source.n), mn(source.mn) {
    flat = source.flat;
  }

  explicit Array(const NdArray<T>& array) {
    GEODE_ASSERT(array.rank()==3);
    m = array.shape[0];
    n = array.shape[1];
    mn = array.shape[2];
    flat = array.flat;
  }

  Array& operator=(const Array& source) {
    flat = source.flat;
    m = source.m;
    n = source.n;
    mn = source.mn;
    return *this;
  }

  // Conversion from mutable to const
  Array& operator=(const MutableSelf& source) {
    flat = source.flat;
    m = source.m;
    n = source.n;
    mn = source.mn;
    return *this;
  }

  template<class TArray> void copy(const TArray& source) {
    resize(source.sizes(),false);
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) for (int ij=0;ij<mn;ij++)
      flat[(i*n+j)*mn+ij] = source(i,j,ij);
  }

  Array<Element,d> copy() const {
    Array<Element,d> result;
    result.copy(*this);
    return result;
  }

  Vector<int,d> sizes() const {
    return Vector<int,d>(m,n,mn);
  }

  void clean_memory() {
    Array empty;
    swap(empty);
  }

  T& operator()(const int i,const int j,const int ij) const {
    assert(unsigned(i)<unsigned(m) && unsigned(j)<unsigned(n) && unsigned(ij)<unsigned(mn));
    return flat[(i*n+j)*mn+ij];
  }

  T& operator()(const Vector<int,d>& index) const {
    assert(unsigned(index.x)<unsigned(m) && unsigned(index.y)<unsigned(n) && unsigned(index.z)<unsigned(mn));
    return flat[(index.x*n+index.y)*mn+index.z];
  }

  RawArray<T> operator()(const int i, const int j) const {
    assert(unsigned(i)<unsigned(m) && unsigned(j)<unsigned(n));
    return RawArray<T>(mn,data()+(i*n+j)*mn);
  }

  RawArray<T,d-1> operator[](const int i) const {
    assert(unsigned(i)<unsigned(m));
    return RawArray<T,d-1>(n,mn,data()+i*n*mn);
  }

  Array<T,d-1> operator()(const int i) const {
    return operator[](i);
  }

  bool valid(const Vector<int,d>& index) const {
    return unsigned(index.x)<unsigned(m) && unsigned(index.y)<unsigned(n) && unsigned(index.z)<unsigned(mn);
  }

  bool valid(const int i,const int j,const int ij) const {
    return unsigned(i)<unsigned(m) && unsigned(j)<unsigned(n) && unsigned(ij)<unsigned(mn);
  }

  void resize(int m_new,int n_new,int mn_new,const bool initialize_new_elements=true,const bool copy_existing_elements=true) {
    if (m_new==m && n_new==n && mn_new==mn) return;
    assert(m_new>=0 && n_new>=0 && mn_new>=0);
    int new_size = m_new*n_new*mn_new;
    Array<T> new_flat(new_size,initialize_new_elements);
    if (copy_existing_elements) {
      int m2 = geode::min(m,m_new),
          n2 = geode::min(n,n_new),
          mn2 = geode::min(mn,mn_new);
      for (int i=0;i<m2;i++) for (int j=0;j<n2;j++) for (int ij=0;ij<mn2;ij++)
        new_flat((i*n_new+j)*mn_new+ij) = flat((i*n+j)*mn+ij);
    }
    m = m_new;
    n = n_new;
    mn = mn_new;
    flat = new_flat;
  }

  void resize(const Vector<int,d>& counts, const bool initialize_new_elements=true, const bool copy_existing_elements=true) {
    resize(counts.x,counts.y,counts.z,initialize_new_elements,copy_existing_elements);
  }

  RawArray<T> reshape(int m_new) const {
    return flat.reshape(m_new);
  }

  RawArray<T,2> reshape(int m_new,int n_new) const {
    return flat.reshape(m_new,n_new);
  }

  void swap(Array& other) {
    flat.swap(other.flat);
    swap(m,other.m);
    swap(n,other.n);
    swap(mn,other.mn);
  }

  RawArray<T,3> slice(int imin,int imax) const {
    assert(unsigned(imin)<=unsigned(imax) && unsigned(imax)<=unsigned(m));
    return RawArray<T,3>(imax-imin,n,mn,data()+imin*n*mn);
  }

  // Extract a subarray at a fixed value of the given axis
  template<int axis> Subarray<T,2> sub(const int i) const {
    static_assert(axis<2,"For now, the last dimension of a Subarray must be contiguous");
    assert(unsigned(i)<unsigned(sizes()[axis]));
    return axis==0 ? (*this)[i] : Subarray<T,2>(m,mn,n*mn,data()+i*mn);
  }
};

template<class T> static inline std::ostream& operator<<(std::ostream& output, const Array<T,3>& a) {
  for (int i=0;i<a.m;i++) {
    for(int j=0;j<a.n;j++) {
      for(int ij=0;ij<a.mn;ij++)
        output<<a(i,j,ij)<<" ";
      output<<std::endl;
    }
    output<<"------------------------------------------"<<std::endl;
  }
  return output;
}

}
