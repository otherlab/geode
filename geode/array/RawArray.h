//#####################################################################
// Class RawArray
//#####################################################################
//
// Version of an array that doesn't own its data, for use in threaded or performance critical code.
// RAW_ARRAYs are returned by default by Slice and other functions (i.e., unless Slice_Own is called).
//
//#####################################################################
#pragma once

#include <geode/array/forward.h>
#include <geode/array/ArrayBase.h>
#include <geode/vector/Vector.h>
#include <geode/python/from_python.h>
#include <geode/structure/forward.h>
#include <geode/utility/HasCheapCopy.h>
#include <geode/utility/CopyConst.h>
#include <geode/utility/range.h>
#include <geode/utility/type_traits.h>
#include <iomanip>
#include <vector>
namespace geode {

template<class T> struct IsArray<RawArray<T> >:public mpl::true_{};
template<class T> struct HasCheapCopy<RawArray<T> >:public mpl::true_{};

template<class T,int d> struct FromPython<RawArray<T,d> >:public FromPython<Array<T,d> >{};

template<class T_>
class RawArray<T_,1> : public ArrayBase<T_,RawArray<T_,1>> {
  typedef T_ T;
  typedef ArrayBase<T,RawArray<T,1>> Base;
public:
  enum Workaroun1 {d=1};
  enum Workaround1 {dimension=d};
  typedef typename remove_const<T>::type Element;
  static const bool is_const = geode::is_const<T>::value;
  typedef T& result_type;

private:
  friend class RawArray<Element>;
  friend class RawArray<const Element>;
  struct Unusable{};

  T* const data_;
public:
  const int m;

  RawArray()
    : data_(0), m(0) {}

  RawArray(const Tuple<>&) // Allow conversion from empty tuples
    : RawArray() {}

  RawArray(const Array<Element>& source)
    : data_(source.data()), m(source.size()) {}

  RawArray(const Array<const Element>& source)
    : data_(source.data()), m(source.size()) {}

  RawArray(const NdArray<T>& array)
    : data_(array.data()), m(array.flat.size()) {
    GEODE_ASSERT(array.rank()==1);
  }

  RawArray(const RawArray<Element>& source)
    : Base(), data_(source.data_), m(source.m) {}

  RawArray(const RawArray<const Element>& source)
    : Base(), data_(source.data_), m(source.m) {}

  RawArray(typename CopyConst<std::vector<Element,std::allocator<Element> >,T>::type& source)
    : Base(), data_(source.size()?&source[0]:0), m((int)source.size()) {}

  RawArray(const int m, T* data)
    : data_(data), m(m) {}

  const RawArray& operator=(const RawArray& source) const {
    assert(size()==source.size());
    memcpy(data_,source.data_,sizeof(T)*m);
    return *this;
  }

  template<class TArray> const RawArray& operator=(const TArray& source) const {
    assert(size()==(int)source.size());
    for (int i=0;i<m;i++) data_[i] = source[i];
    return *this;
  }

  int size() const {
    return m;
  }

  Vector<int,1> sizes() const {
    return Vector<int,1>(m);
  }

  T& operator()(const int i) const {
    assert(unsigned(i)<unsigned(m));
    return data_[i];
  }

  T& operator[](const int i) const {
    assert(unsigned(i)<unsigned(m));
    return data_[i];
  }

  bool valid(const int i) const {
    return unsigned(i)<unsigned(m);
  }

  T* data() const {
    return data_;
  }

  void zero() const {
    static_assert(IsScalarVectorSpace<T>::value,"");
    memset(data_,0,m*sizeof(T));
  }

  Vector<int,1> index(const int i) const {
    return vec(i);
  }

  RawArray slice(int lo, int hi) const {
    assert(unsigned(lo)<=unsigned(hi) && unsigned(hi)<=unsigned(m));
    return RawArray(hi-lo,data_+lo);
  }

  RawArray slice(Range<int> range) const {
    return slice(range.lo,range.hi);
  }

  RawArray<T,2> reshape(int m_new,int n_new) const {
    assert(m_new*n_new==m);
    return RawArray<T,2>(m_new,n_new,data_);
  }

  RawArray<T,3> reshape(int m_new,int n_new,int mn_new) const {
    assert(m_new*n_new*mn_new==m);
    return RawArray<T,3>(m_new,n_new,mn_new,data_);
  }

  RawArray<Element> const_cast_() const {
    return RawArray<Element>(m,const_cast<Element*>(data_));
  }

  RawArray<const Element> const_() const {
    return *this;
  }

  T* begin() const { // for stl
    return data_;
  }

  T* end() const { // for stl
    return data_+m;
  }
};

template<class T>
class RawArray<T,2> {
public:
  enum Workaroun1 {d=2};
  enum Workaround1 {dimension=d};
  typedef typename remove_const<T>::type Element;
  static const bool is_const = geode::is_const<T>::value;

private:
  friend class RawArray<typename mpl::if_c<is_const,Element,const Element>::type,2>;
  struct Unusable{};

public:
  RawArray<T> flat;
  int m,n;

  RawArray(const Array<Element,2>& source)
    : flat(source.flat), m(source.m), n(source.n)
  {}

  RawArray(const Array<const Element,2>& source)
    : flat(source.flat), m(source.m), n(source.n)
  {}

  explicit RawArray(const NdArray<T>& array)
    : flat(array.flat) {
    GEODE_ASSERT(array.rank()==2);
    m = array.shape[0];
    n = array.shape[1];
  }

  RawArray(const RawArray<Element,2>& source)
    : flat(source.flat), m(source.m), n(source.n) {}

  RawArray(const RawArray<const Element,2>& source)
    : flat(source.flat), m(source.m), n(source.n) {}

  RawArray(int m, int n, T* data)
    : flat(m*n,data), m(m), n(n) {
    assert(m>=0 && n>=0);
  }

  const RawArray& operator=(const RawArray& source) const {
    assert(sizes()==source.sizes());
    flat = source.flat;
    return *this;
  }

  template<class TArray> const RawArray& operator=(const TArray& source) const {
    assert(sizes()==source.sizes());
    int k = 0;
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) flat[k++] = source(i,j);
    return *this;
  }

  Array<Element,2> copy() const {
    Array<Element,2> copy;
    copy.copy(*this);
    return copy;
  }

  Vector<int,2> sizes() const {
    return Vector<int,2>(m,n);
  }

  int total_size() const {
    return flat.size();
  }

  T& operator()(const int i,const int j) const {
    assert(unsigned(i)<unsigned(m) && unsigned(j)<unsigned(n));
    return flat[i*n+j];
  }

  T& operator[](const Vector<int,2>& index) const {
    assert(unsigned(index.x)<unsigned(m) && unsigned(index.y)<unsigned(n));
    return flat[index.x*n+index.y];
  }

  RawArray<T> operator[](const int i) const {
    assert(unsigned(i)<unsigned(m));return RawArray<T>(n,data()+i*n);
  }

  Vector<int,2> index(const int i) const {
    int x = i/n;
    return vec(x, i-x);
  }

  RawArray<T> row(int i) const {
    return (*this)[i];
  }

  Subarray<T> column(int j) const {
    return Subarray<T>(flat,j,flat.size(),n);
  }

  RawArray<T> back() const {
    return (*this)[m-1];
  }

  bool valid(const Vector<int,2>& index) const {
    return unsigned(index.x)<unsigned(m) && unsigned(index.y)<unsigned(n);
  }

  bool valid(const int i,const int j) const {
    return unsigned(i)<m && unsigned(j)<n;
  }

  const RawArray& operator+=(const RawArray& v) const {
    assert(sizes()==v.sizes());
    flat += v.flat;
    return *this;
  }

  const RawArray& operator-=(const RawArray& v) const {
    assert(sizes()==v.sizes());
    flat -= v.flat;
    return *this;
  }

  const RawArray& operator*=(const RawArray& v) const {
    assert(sizes()==v.sizes());
    flat *= v.flat;
    return *this;
  }

  const RawArray& operator/=(const RawArray& v) const {
    assert(sizes()==v.sizes());
    flat /= v.flat;
    return *this;
  }

  void fill(const T& constant) const {
    flat.fill(constant);
  }

  template<class T2> RawArray& operator*=(const T2& a) {
    flat *= a;
    return *this;
  }

  T* data() const {
    return flat.data();
  }

  T maxabs() const {
    return flat.maxabs();
  }

  Vector<int,2> argmaxabs() const {
    int i = flat.argmaxabs();
    return Vector<int,2>(i/n,i%n);
  }

  RawArray slice(int imin,int imax) const {
    assert(unsigned(imin)<=unsigned(imax) && unsigned(imax)<=unsigned(m));
    return RawArray(imax-imin,n,data()+imin*n);
  }

  const Subarray<T,2> slice(int imin,int imax,int jmin,int jmax) const {
    return Subarray<T,2>(*this,imin,imax,jmin,jmax);
  }

  const Subarray<T> diagonal() const {
    return Subarray<T>(flat,0,min(m,n)*(n+1)-n,n+1);
  }

  RawArray<T,2> reshape(int m_new,int n_new) const {
    return flat.reshape(m_new,n_new);
  }

  template<int k> RawArray<typename mpl::if_c<is_const,const Vector<Element,k>,Vector<T,k> >::type> vector_view() const {
    GEODE_ASSERT(n==k);typedef typename mpl::if_c<is_const,const Vector<Element,k>,Vector<T,k> >::type TV;
    return RawArray<TV>(m,(TV*)data());
  }

  RawArray<const Element,2> const_() const {
    return *this;
  }

  GEODE_CORE_EXPORT void transpose();
  GEODE_CORE_EXPORT Array<Element,2> transposed() const;
  GEODE_CORE_EXPORT void permute_rows(RawArray<const int> p,int direction) const; // 1 for forward (A[i] = A[p[i]]), -1 for backward (A[p[i]] = A[i])
  GEODE_CORE_EXPORT void permute_columns(RawArray<const int> p,int direction) const; // 1 for forward (A[i] = A[p[i]]), -1 for backward (A[p[i]] = A[i])
};

GEODE_CORE_EXPORT real frobenius_norm(RawArray<const real,2> A);
GEODE_CORE_EXPORT real infinity_norm(RawArray<const real,2> A); // Matrix infinity norm

template<class T>
class RawArray<T,3> {
public:
  enum Workaroun1 {d=3};
  enum Workaround1 {dimension=d};
  typedef typename remove_const<T>::type Element;
  static const bool is_const = geode::is_const<T>::value;

private:
  friend class RawArray<typename mpl::if_c<is_const,Element,const Element>::type,3>;
  struct Unusable{};

public:
  RawArray<T> flat;
  int m,n,mn;

  RawArray(const Array<Element,3>& source)
    : flat(source.flat), m(source.m), n(source.n), mn(source.mn)
  {}

  RawArray(const Array<const Element,3>& source)
    : flat(source.flat), m(source.m), n(source.n), mn(source.mn)
  {}

  RawArray(const RawArray<Element,3>& source)
    : flat(source.flat), m(source.m), n(source.n), mn(source.mn)
  {}

  RawArray(const RawArray<const Element,3>& source)
    : flat(source.flat), m(source.m), n(source.n), mn(source.mn)
  {}

  RawArray(int m, int n, int mn, T* data)
    : flat(m*n*mn,data), m(m), n(n), mn(mn) {
    assert(m>=0 && n>=0 && mn>=0);
  }

  RawArray& operator=(const RawArray& source) const {
    assert(sizes()==source.sizes());
    flat = source.flat;
  }

  template<class TArray> RawArray& operator=(const TArray& source) const {
    assert(sizes()==source.sizes());
    int k = 0;
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) for (int ij=0;ij<mn;ij++)
      flat[k++] = source(i,j,ij);
  }

  Array<Element,3> copy() const {
    Array<Element,3> copy;
    copy.copy(*this);
    return copy;
  }

  Vector<int,3> sizes() const {
    return Vector<int,3>(m,n,mn);
  }

  int total_size() const {
    return flat.size();
  }

  T& operator()(const int i,const int j,const int ij) const {
    assert(unsigned(i)<unsigned(m) && unsigned(j)<unsigned(n) && unsigned(ij)<unsigned(mn));
    return flat[(i*n+j)*mn+ij];
  }

  T& operator[](const Vector<int,3>& index) const {
    return operator()(index.x,index.y,index.z);
  }

  RawArray<T> operator()(const int i, const int j) const {
    assert(unsigned(i)<unsigned(m) && unsigned(j)<unsigned(n));
    return RawArray<T>(mn,data()+(i*n+j)*mn);
  }

  RawArray<T,2> operator[](int i) const {
    assert(unsigned(i)<unsigned(m));
    return RawArray<T,2>(n,mn,data()+i*n*mn);
  }

  bool valid(const int i,const int j,const int ij) const {
    return unsigned(i)<unsigned(m) && unsigned(j)<unsigned(n) && unsigned(ij)<unsigned(mn);
  }

  bool valid(const Vector<int,3>& index) const {
    return valid(index.x,index.y,index.z);
  }

  void fill(const T& constant) const {
    flat.fill(constant);
  }

  template<class T2> RawArray& operator*=(const T2& a) {
    flat *= a;
    return *this;
  }

  T* data() const {
    return flat.data();
  }

  Vector<int,3> argmaxabs() const {
    int k = flat.argmaxabs(),
        ij = k%mn;
    k /= mn;
    return Vector<int,3>(k/n,k%n,ij);
  }

  RawArray slice(int imin,int imax) const {
    assert(unsigned(imin)<=unsigned(imax) && unsigned(imax)<=unsigned(m));
    return RawArray(imax-imin,n,mn,data()+imin*n*mn);
  }

  // Extract a subarray at a fixed value of the given axis
  template<int axis> Subarray<T,2> sub(const int i) const {
    static_assert(axis<2,"For now, the last dimension of a Subarray must be contiguous");
    assert(unsigned(i)<unsigned(sizes()[axis]));
    return axis==0 ? (*this)[i] : Subarray<T,2>(m,mn,n*mn,data()+i*mn);
  }

  RawArray<const Element,3> const_() const {
    return *this;
  }
};

template<class T>
class RawArray<T,4> {
public:
  enum Workaroun1 {d=4};
  enum Workaround1 {dimension=d};
  typedef typename remove_const<T>::type Element;
  static const bool is_const = geode::is_const<T>::value;

private:
  friend class RawArray<typename mpl::if_c<is_const,Element,const Element>::type,4>;
  struct Unusable{};

public:
  RawArray<T> flat;
  Vector<int,4> shape;

  RawArray(const Array<Element,4>& source)
    : flat(source.flat), shape(source.shape) {}

  RawArray(const Array<const Element,4>& source)
    : flat(source.flat), shape(source.shape) {}

  RawArray(const RawArray<Element,4>& source)
    : flat(source.flat), shape(source.shape) {}

  RawArray(const RawArray<const Element,4>& source)
    : flat(source.flat), shape(source.shape) {}

  RawArray(const Vector<int,4>& shape, T* data)
    : flat(shape.product(),data), shape(shape) {}

  RawArray& operator=(const RawArray& source) const {
    assert(sizes()==source.sizes());
    flat = source.flat;
  }

  template<class TArray> RawArray& operator=(const TArray& source) const {
    assert(sizes()==source.sizes());
    int f=0;
    for (int i=0;i<shape[0];i++) for(int j=0;j<shape[1];j++) for(int k=0;k<shape[2];k++) for(int l=0;l<shape[3];l++)
      flat[f++] = source(i,j,k,l);
  }

  Array<Element,4> copy() const {
    Array<Element,4> copy;
    copy.copy(*this);
    return copy;
  }

  const Vector<int,4>& sizes() const {
    return shape;
  }

  int total_size() const {
    return flat.size();
  }

  T& operator()(const int i,const int j,const int k,const int l) const {
    assert(unsigned(i)<unsigned(shape[0]) && unsigned(j)<unsigned(shape[1]) && unsigned(k)<unsigned(shape[2]) && unsigned(l)<unsigned(shape[3]));
    return flat[((i*shape[1]+j)*shape[2]+k)*shape[3]+l];
  }

  T& operator[](const Vector<int,4>& index) const {
    return operator()(index.x,index.y,index.z,index.w);
  }

  bool valid(const int i,const int j,const int k,const int l) const {
    return unsigned(i)<unsigned(shape[0]) && unsigned(j)<unsigned(shape[1]) && unsigned(k)<unsigned(shape[2]) && unsigned(l)<unsigned(shape[3]);
  }

  bool valid(const Vector<int,4>& index) const {
    return valid(index.x,index.y,index.z,index.w);
  }

  void fill(const T& constant) const {
    flat.fill(constant);
  }

  template<class T2> RawArray& operator*=(const T2& a) {
    flat *= a;
    return *this;
  }

  T* data() const {
    return flat.data();
  }

  RawArray<const Element,4> const_() const {
    return *this;
  }
};

template<class T, int d> static inline RawArray<T,d> &flat(RawArray<T,d> &A) {
  return A.flat;
}

template<class T> static inline RawArray<T,1> &flat(RawArray<T,1> &A) {
  return A;
}

template<class T, int d> static inline RawArray<T,d> const &flat(RawArray<T,d> const &A) {
  return A.flat;
}

template<class T> static inline RawArray<T,1> const &flat(RawArray<T,1> const &A) {
  return A;
}

template<class T> static inline std::ostream& operator<<(std::ostream& output,const RawArray<T,2>& a) {
    return output<<Subarray<const T,2>(a);
}

struct TemporaryOwner {
  Ref<PyObject> owner;
  GEODE_CORE_EXPORT TemporaryOwner(); // creates owner with reference count 1
  GEODE_CORE_EXPORT ~TemporaryOwner(); // bails if reference count isn't back to 1

  template<class T,int d> Array<T,d> share(RawArray<T,d> array) {
    return Array<T,d>(array.sizes(),array.data(),&*owner);
  }
};

}
