//#####################################################################
// Class Array<T,4>
//#####################################################################
//
// A shareable, four-dimensional array.
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
class Array<T_,4> : public ArrayNdBase<T_,Array<T_,4> > {
  typedef T_ T;
public:
  enum Workaround1 {dimension=4};
  enum Workaround2 {d=dimension};
  typedef typename remove_const<T>::type Element;
  typedef ArrayNdBase<T,Array> Base;

  using Base::flat; // one-dimensional data storage
  using Base::data;

private:
  struct Unusable{};
  typedef typename mpl::if_<is_const<T>,Array<Element,d>,Unusable>::type MutableSelf;
public:

  Vector<int,4> shape;

  Array() {}

  Array(const Vector<int,d> shape)
    : Base((assert(shape.min()>=0),
            shape.x*shape.y*shape.z*shape.w))
    , shape(shape) {}

  Array(const Vector<int,d> shape, Uninit)
    : Base((assert(shape.min()>=0),
            shape.x*shape.y*shape.z*shape.w),uninit)
    , shape(shape) {}

  Array(const Vector<int,d> shape, T* data, PyObject* owner)
    : Base((assert(shape.min()>=0),
            shape.product()),data,owner)
    , shape(shape) {}

  Array(const Array& source)
    : Base(source.flat), shape(source.shape) {}

  // Conversion from mutable to const
  Array(const MutableSelf& source)
    : Base(source.flat), shape(source.shape) {}

  Array& operator=(const Array& source) {
    flat = source.flat;
    shape = source.shape;
    return *this;
  }

  // Conversion from mutable to const
  Array& operator=(const MutableSelf& source) {
    flat = source.flat;
    shape = source.shape;
    return *this;
  }

  template<class TArray> void copy(const TArray& source) {
    if ((void*)this == (void*)&source)
      return;
    clear();
    resize(source.sizes(),uninit);
    int f = 0;
    for (int i=0;i<shape[0];i++)
      for (int j=0;j<shape[1];j++)
        for (int k=0;k<shape[2];k++)
          for (int l=0;l<shape[3];l++)
            flat[f++] = source(i,j,k,l);
  }

  Array<Element,d> copy() const {
    Array<Element,d> result;
    result.copy(*this);
    return result;
  }

  const Vector<int,d>& sizes() const {
    return shape;
  }

  void clear() {
    flat.clear();
    shape.fill(0);
  }

  void clean_memory() {
    Array empty;
    swap(empty);
  }

  T& operator()(const int i,const int j,const int k,const int l) const {
    assert(valid(i,j,k,l));
    return flat[((i*shape[1]+j)*shape[2]+k)*shape[3]+l];
  }

  T& operator[](const Vector<int,d>& index) const {
    return (*this)(index.x,index.y,index.z,index.w);
  }

  bool valid(const int i,const int j,const int k,const int l) const {
    return unsigned(i)<unsigned(shape[0]) && unsigned(j)<unsigned(shape[1]) && unsigned(k)<unsigned(shape[2]) && unsigned(l)<unsigned(shape[3]);
  }

  bool valid(const Vector<int,d>& index) const {
    return valid(index.x,index.y,index.z,index.w);
  }

  void resize(const Vector<int,d>& shape_new) {
    if (shape_new==shape) return;
    assert(shape_new.min()>=0);
    int new_size = shape_new.product();
    Array<T> new_flat(new_size);
    const Vector<int,d> common = Vector<int,d>::componentwise_min(shape,shape_new);
    for (int i=0;i<common[0];i++) for (int j=0;j<common[1];j++) for (int k=0;k<common[2];k++) for (int l=0;l<common[3];l++)
      new_flat[((i*shape_new[1]+j)*shape_new[2]+k)*shape_new[3]+l] = flat[((i*shape[1]+j)*shape[2]+k)*shape[3]+l];
    shape = shape_new;
    flat = new_flat;
  }

  void resize(const Vector<int,d>& shape_new, Uninit) {
    if (shape_new==shape) return;
    assert(shape_new.min()>=0);
    int new_size = shape_new.product();
    Array<T> new_flat(new_size,uninit);
    const Vector<int,d> common = Vector<int,d>::componentwise_min(shape,shape_new);
    for (int i=0;i<common[0];i++) for (int j=0;j<common[1];j++) for (int k=0;k<common[2];k++) for (int l=0;l<common[3];l++)
      new_flat[((i*shape_new[1]+j)*shape_new[2]+k)*shape_new[3]+l] = flat[((i*shape[1]+j)*shape[2]+k)*shape[3]+l];
    shape = shape_new;
    flat = new_flat;
  }

  RawArray<T> reshape(int m_new) const {
    return flat.reshape(m_new);
  }

  RawArray<T,2> reshape(int m_new,int n_new) const {
    return flat.reshape(m_new,n_new);
  }

  void swap(Array& other) {
    flat.swap(other.flat);
    swap(shape,other.shape);
  }
};

}
