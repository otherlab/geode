//#####################################################################
// Class Subarray
//#####################################################################
//
// A slice of an Array<T,d>.  SUBARRAYs never own their data, so treat with care.
//
//#####################################################################
#pragma once

#include <geode/array/forward.h>
#include <geode/vector/Vector.h>
#include <geode/utility/str.h>
#include <iomanip>
namespace geode {

template<class T_>
class Subarray<T_,1> : public ArrayBase<T_,Subarray<T_,1> > {
  typedef T_ T;
public:
  enum Workaroun1 {d=1};
  enum Workaround1 {dimension=d};
  typedef typename boost::remove_const<T>::type Element;
  static const bool is_const=boost::is_const<T>::value;

private:
  friend class Subarray<Element,2>;
  friend class Subarray<const Element,2>;
  struct Unusable{};

  T* const data_; // raw data, with gaps if stride > 1
public:
  const int m;
  const int stride; // stride of outer dimension (inner is assumed contiguous)

  template<class T2>
  Subarray(const Array<T2>& source, typename boost::enable_if<mpl::or_<boost::is_same<T2,T>,boost::is_same<T2,Element> >,Unusable>::type unusable=Unusable())
    : data_(source.data()), m(source.size()), stride(1) {}

  template<class T2>
  Subarray(const RawArray<T2>& source, typename boost::enable_if<mpl::or_<boost::is_same<T2,T>,boost::is_same<T2,Element> >,Unusable>::type unusable=Unusable())
    : data_(source.data()), m(source.size()), stride(1) {}

  template<class T2>
  Subarray(const Array<T2>& source, int lo, int hi, int stride)
    : data_(source.data()+lo), m((hi-lo-1+stride)/stride), stride(stride) {
    assert(unsigned(lo)<=unsigned(hi+stride-1) && unsigned(hi)<=unsigned(source.size()));
    assert(data_ || !m);
  }

  template<class T2>
  Subarray(const RawArray<T2>& source,int lo,int hi,int stride)
    : data_(source.data()+lo), m((hi-lo-1+stride)/stride), stride(stride) {
    assert(unsigned(lo)<=unsigned(hi+stride-1) && unsigned(hi)<=unsigned(source.size()));
    assert(data_ || !m);
  }

  Subarray(const Subarray& source)
    : data_(source.data()), m(source.m), stride(source.stride) {}

  Subarray(typename mpl::if_c<is_const,const Subarray<Element>&,Unusable>::type source)
    : data_(source.data()), m(source.m), stride(source.stride) {}

  const Subarray& operator=(const Subarray& source) const {
    assert(size()==source.size());
    for (int i=0;i<m;i++) data_[i*stride] = source(i);
    return *this;
  }

  template<class TArray> const Subarray& operator=(const TArray& source) const {
    assert(size()==source.size());
    for (int i=0;i<m;i++) data_[i*stride] = source[i];
    return *this;
  }

  int size() const {
    return m;
  }

  T& operator()(const int i) const {
    assert(unsigned(i)<unsigned(m));
    return data_[i*stride];
  }

  T& operator[](const int i) const {
    return operator()(i);
  }

  bool valid(const int i) const {
    return unsigned(i)<unsigned(m);
  }

  // Warning: data is noncontiguous if stride>1
  T* data() const {
    return data_;
  }
};

template<class T>
class Subarray<T,2> {
public:
  enum Workaroun1 {d=2};
  enum Workaround1 {dimension=d};
  typedef typename boost::remove_const<T>::type Element;
  static const bool is_const=boost::is_const<T>::value;

private:
  friend class Subarray<typename mpl::if_c<is_const,Element,const Element>::type,2>;
  struct Unusable{};

  T* const data_; // raw data, with gaps if stride > 1
public:
  const int m,n;
  const int stride; // stride of outer dimension (inner is assumed contiguous)

  Subarray(const Array<Element,2>& source)
    : data_(source.data()), m(source.m), n(source.n), stride(n) {}

  Subarray(RawArray<Element,2> source)
    : data_(source.data()), m(source.m), n(source.n), stride(n) {}

  Subarray(const Array<const Element,2>& source)
    : data_(source.data()), m(source.m), n(source.n), stride(n) {}

  Subarray(RawArray<const Element,2> source)
    : data_(source.data()), m(source.m), n(source.n), stride(n) {}

  Subarray(Array<T,2> source, int ilo, int ihi, int jlo, int jhi)
    : data_(source.data()+ilo*source.n+jlo), m(ihi-ilo), n(jhi-jlo), stride(source.n) {
    assert(unsigned(ilo)<=unsigned(ihi) && unsigned(ihi)<=unsigned(source.m) && unsigned(jlo)<=unsigned(jhi) && unsigned(jhi)<=unsigned(source.n));
    assert(data_ || !m || !n);
  }

  Subarray(RawArray<T,2> source, int ilo, int ihi, int jlo, int jhi)
    : data_(source.data()+ilo*source.n+jlo), m(ihi-ilo), n(jhi-jlo), stride(source.n) {
    assert(unsigned(ilo)<=unsigned(ihi) && unsigned(ihi)<=unsigned(source.m) && unsigned(jlo)<=unsigned(jhi) && unsigned(jhi)<=unsigned(source.n));
    assert(data_ || !m || !n);
  }

  Subarray(const Subarray& source)
    : data_(source.data()), m(source.m), n(source.n), stride(source.stride) {}

  Subarray(typename mpl::if_c<is_const,const Subarray<Element,2>&,Unusable>::type source)
    : data_(source.data()), m(source.m), n(source.n), stride(source.stride) {}

  Subarray(const int m, const int n, const int stride, T* data_)
    : data_(data_), m(m), n(n), stride(stride) {}

  const Subarray& operator=(const Subarray& source) const {
    assert(sizes()==source.sizes());
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) data_[i*stride+j] = source(i,j);
    return *this;
  }

  template<class TArray> const Subarray& operator=(const TArray& source) const {
    assert(sizes()==source.sizes());
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) data_[i*stride+j] = source(i,j);
    return *this;
  }

  Vector<int,2> sizes() const {
    return Vector<int,2>(m,n);
  }

  T& operator()(const int i,const int j) const {
    assert(unsigned(i)<unsigned(m) && unsigned(j)<unsigned(n));
    return data_[i*stride+j];
  }

  T& operator()(const Vector<int,2>& index) const {
    assert(unsigned(index.x)<unsigned(m) && unsigned(index.y)<unsigned(n));
    return data_[index.x*stride+index.y];
  }

  RawArray<T> operator[](const int i) const {
    assert(unsigned(i)<unsigned(m));
    return RawArray<T>(n,data_+i*stride);
  }

  bool valid(const Vector<int,2>& index) const {
    return unsigned(index.x)<unsigned(m) && unsigned(index.y)<unsigned(n);
  }

  bool valid(const int i,const int j) const {
    return unsigned(i)<unsigned(m) && unsigned(j)<unsigned(n);
  }

  void fill(const T& constant) const {
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) data_[i*stride+j] = constant;
  }

  template<class T2> typename boost::enable_if<mpl::or_<IsScalar<T2>,boost::is_same<T,T2> >,Subarray&>::type operator*=(const T2& a) {
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) data_[i*stride+j] *= a;
    return *this;
  }

  // Warning: data is noncontiguous if stride>1
  T* data() const {
    return data_;
  }

  Vector<int,2> argmaxabs() const {
    assert(m || n);
    Vector<int,2> ij;
    T a = abs(data_[0]);
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) {
      T b = abs(data_[i*stride+j]);
      if (a<b) {
        a = b;
        ij.set(i,j);
      }
    }
    return ij;
  }

  Array<T,2> copy() const {
    Array<T,2> copy(m,n,false);
    for (int i=0;i<m;i++) for (int j=0;j<n;j++)
      copy(i,j) = (*this)(i,j);
    return copy;
  }
};

template<class T> inline std::ostream& operator<<(std::ostream& output, const Subarray<T,1>& a) {
  for (int i=0;i<a.m;i++) output<<a[i]<<' ';
  return output<<std::endl;
}

template<class T> inline std::ostream& operator<<(std::ostream& output,const Subarray<T,2>& a) {
  int width=0;
  for (int i=0;i<a.m;i++) for (int j=0;j<a.n;j++)
    width = max(width,str(a(i,j)).size());
  for (int i=0;i<a.m;i++) {
    for (int j=0;j<a.n;j++)
      output<<std::setw(width)<<str(a(i,j))<<' ';
    output<<std::endl;
  }
  return output;
}

}
