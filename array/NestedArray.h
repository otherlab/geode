//#####################################################################
// Class NestedArray
//#####################################################################
//
// NestedArray<T> is essentially Array<Array<T>>, but stored in flat form for efficiency.
// Like Array<T>, nested arrays are shareable, and are exposed to python via sharing instead
// of as a native datatype.  See __init__.py for the python version.
//
//#####################################################################
#pragma once

#include <other/core/array/Array.h>
namespace other {

template<class T> OTHER_CORE_EXPORT PyObject* to_python(const NestedArray<T>& array);
template<class T> struct FromPython<NestedArray<T> >{OTHER_CORE_EXPORT static NestedArray<T> convert(PyObject* object);};
OTHER_CORE_EXPORT Array<int> nested_array_offsets(RawArray<const int> lengths);

template<class T>
class NestedArray {
public:
  typedef typename Array<T>::Element Element;

  Array<const int> offsets;
  Array<T> flat;

  NestedArray()
    : offsets(nested_array_offsets(Array<const int>())) {}

  NestedArray(RawArray<const int> lengths, bool initialize=true)
    : offsets(nested_array_offsets(lengths)), flat(offsets.back(),initialize) {}

  template<class S> NestedArray(const NestedArray<S>& other) {
    *this = other;
  }

  // Note: To convert vector<vector<T>> to a NestedArray, use copy below

  template<class S> NestedArray& operator=(const NestedArray<S>& other) {
    offsets = other.offsets;
    flat = other.flat;
    return *this;
  }

  template<class S> static NestedArray zeros_like(const NestedArray<S>& other) {
    NestedArray array;
    array.offsets = other.offsets;
    array.flat.resize(array.offsets.back());
    return array;
  }

  template<class S> static NestedArray empty_like(const NestedArray<S>& other) {
    NestedArray array;
    array.offsets = other.offsets;
    array.flat.resize(array.offsets.back(),false,false);
    return array;
  }

  template<class S> static NestedArray reshape_like(Array<T> flat,const NestedArray<S>& other) {
    OTHER_ASSERT(other.flat.size()==flat.size());
    NestedArray array;
    array.offsets = other.offsets;
    array.flat = flat;
    return array;
  }

  template<class TA> static NestedArray copy(const TA& other) {
    const int n = (int)other.size();
    Array<int> offsets(n+1,false);
    offsets[0] = 0;
    for (int i=0;i<n;i++)
      offsets[i+1] = offsets[i]+(int)other[i].size();
    Array<Element> flat(offsets[n],false);
    for (int i=0;i<n;i++)
      flat.slice(offsets[i],offsets[i+1]) = other[i];
    NestedArray self;
    self.offsets = offsets;
    self.flat = flat;
    return self;
  }

  int size() const {
    return offsets.size()-1;
  }

  int size(int i) const {
    return offsets[i+1]-offsets[i];
  }

  bool valid(int i) const {
    return unsigned(i)<unsigned(size());
  }

  int total_size() const {
    return offsets.back();
  }

  Array<int> sizes() const {
    return (offsets.slice(1,offsets.size())-offsets.slice(0,offsets.size()-1)).copy();
  }

  T& operator()(int i,int j) const {
    int index = offsets[i]+j;
    assert(0<=j && index<=offsets[i+1]);
    return flat[index];
  }

  RawArray<T> operator[](int i) const {
    return flat.slice(offsets[i],offsets[i+1]);
  }

  NestedArray<Element> copy() const {
    NestedArray<Element> copy;
    copy.offsets = offsets;
    copy.flat = flat.copy();
    return copy;
  }

  ArrayIter<NestedArray> begin() const {
    return ArrayIter<NestedArray>(*this,0);
  }

  ArrayIter<NestedArray> end() const {
    return ArrayIter<NestedArray>(*this,size());
  }
};

}
