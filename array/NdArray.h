//#####################################################################
// Class NdArray
//#####################################################################
//
// NdArray represents a flat, shareable, arbitrary rank array, primarily for interaction with Python.
//
//#####################################################################
#pragma once

#include <other/core/array/Array.h>
namespace other {

using std::ostream;
template<class T> OTHER_CORE_EXPORT PyObject* to_python(const NdArray<T>& array);
template<class T> struct FromPython<NdArray<T> >{OTHER_CORE_EXPORT static NdArray<T> convert(PyObject* object);};

template<class T>
class NdArray {
public:
  typedef typename boost::remove_const<T>::type Element;
  static const bool is_const=boost::is_const<T>::value;

  Array<const int> shape;
  Array<T> flat;

  NdArray()
    : flat(Array<Element>(1)) {}

  NdArray(Array<const int> shape, const bool initialize=true)
    : shape(shape), flat(shape.product(),initialize) {
    assert(!shape.size() || shape.min()>=0);
  }

  NdArray(Array<const int> shape, T* data, PyObject* owner)
    : shape(shape), flat(shape.product(),data,owner) {
    assert(!shape.size() || shape.min()>=0);
  }

  NdArray(const NdArray<Element>& source)
    : shape(source.shape), flat(source.flat) {}

  NdArray(const NdArray<const Element>& source)
    : shape(source.shape), flat(source.flat) {}

  NdArray(const Array<T>& array)
    : flat(array) {
    Array<int> shape_;
    shape_.copy(array.sizes());
    shape = shape_;
  }

  template<int d> NdArray(const Array<T,d>& array)
    : flat(array.flat) {
    Array<int> shape_;
    shape_.copy(array.sizes());
    shape = shape_;
  }

  template<int m> NdArray(const Array<Vector<T,m> >& array)
    : flat(m*array.size(),reinterpret_cast<T*>(array.data()),array.borrow_owner()) {
    Array<int> shape_;
    shape_.copy(vec(array.size(),m));
    shape = shape_;
  }

  NdArray& operator=(const NdArray<Element>& source) {
    shape = source.shape;
    flat = source.flat;
    return *this;
  }

  NdArray& operator=(const NdArray<const Element>& source) {
    shape = source.shape;
    flat = source.flat;
    return *this;
  }

  int rank() const {
    return shape.size();
  }

  PyObject* owner() const {
    return flat.owner();
  }

  T* data() const {
    return flat.data();
  }

  void swap(NdArray& other) {
    shape.swap(other.shape);
    flat.swap(other.flat);
  }

  template<int d> Array<T,d> as() const {
    OTHER_ASSERT(rank()==d);
    Vector<int,d> shape_;
    for (int i=0;i<d;i++) shape_[i] = shape[i];
    return Array<T,d>(shape_,flat.data(),flat.borrow_owner());
  }

  T& operator()() const {
    assert(rank()==0);
    return flat[0];
  }

  T& operator()(int i) const {
    assert(rank()==1 && unsigned(i)<unsigned(shape[0]));
    return flat[i];
  }

  T& operator()(int i,int j) const {
    assert(rank()==2 && unsigned(i)<unsigned(shape[0]) && unsigned(j)<unsigned(shape[1]));
    return flat[i*shape[1]+j];
  }

  T& operator()(int i,int j,int k) const {
    assert(rank()==3 && unsigned(i)<unsigned(shape[0]) && unsigned(j)<unsigned(shape[1]) && unsigned(k)<unsigned(shape[2]));
    return flat[(i*shape[1]+j)*shape[2]+k];
  }

  T& operator()(int i,int j,int k, int l) const {
    assert(rank()==4 && unsigned(i)<unsigned(shape[0]) && unsigned(j)<unsigned(shape[1]) && unsigned(k)<unsigned(shape[2]) && unsigned(l)<unsigned(shape[3]));
    return flat[((i*shape[1]+j)*shape[2]+k)*shape[3]+l];
  }

  T& operator[](int i) const {
    return (*this)(i);
  }

  T& operator[](const Vector<int,0>& I) const {
    return (*this)();
  }

  T& operator[](const Vector<int,1>& I) const {
    return (*this)(I.x);
  }

  T& operator[](const Vector<int,2>& I) const {
    return (*this)(I.x,I.y);
  }

  T& operator[](const Vector<int,3>& I) const {
    return (*this)(I.x,I.y,I.z);
  }

  T& operator[](const Vector<int,4>& I) const {
    return (*this)(I.x,I.y,I.z,I.w);
  }
};

template<class T> inline ostream& operator<<(ostream& output, const NdArray<T>& a) {
  return output << "NdArray(shape=" << a.shape << ",flat=" << a.flat << ')';
}

}
