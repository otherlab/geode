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

using std::ostream;
OTHER_CORE_EXPORT bool is_nested_array(PyObject* object);
OTHER_CORE_EXPORT Array<int> nested_array_offsets(RawArray<const int> lengths);

template<class T,bool frozen> // frozen=true
class NestedArray {
  struct Unusable {};
public:
  typedef typename Array<T>::Element Element;
  template<class S,bool f> struct Compatible { static const bool value = boost::is_same<Element,typename boost::remove_const<S>::type>::value && boost::is_const<T>::value>=boost::is_const<S>::value; };

  // When growing an array incrementally via append or extend, set frozen=false to make offsets mutable.
  typedef Array<typename mpl::if_c<frozen,const int,int>::type> Offsets;

  Offsets offsets;
  Array<T> flat;

  NestedArray()
    : offsets(nested_array_offsets(Array<const int>())) {}

  NestedArray(RawArray<const int> lengths, bool initialize=true)
    : offsets(nested_array_offsets(lengths)), flat(offsets.back(),initialize) {}

  NestedArray(const NestedArray& other)
    : offsets(other.offsets)
    , flat(other.flat) {}

  template<class S,bool f> NestedArray(const NestedArray<S,f>& other, typename boost::enable_if<Compatible<S,f>,Unusable>::type unusable=Unusable())
    : offsets(other.offsets)
    , flat(other.flat) {}

  // Build from manually constructed offsets and flat arrays
  NestedArray(const Offsets& offsets, const Array<T>& flat)
    : offsets(offsets), flat(flat) {
    assert(offsets.back()==flat.size());
  }

  // Note: To convert vector<vector<T>> to a NestedArray, use copy below

  template<class S,bool f> NestedArray& operator=(const NestedArray<S,f>& other) {
    offsets = other.offsets;
    flat = other.flat;
    return *this;
  }

  template<class S,bool f> static NestedArray zeros_like(const NestedArray<S,f>& other) {
    NestedArray array;
    array.offsets = other.offsets;
    array.flat.resize(array.offsets.back());
    return array;
  }

  template<class S,bool f> static NestedArray empty_like(const NestedArray<S,f>& other) {
    NestedArray array;
    array.offsets = other.offsets;
    array.flat.resize(array.offsets.back(),false,false);
    return array;
  }

  template<class S,bool f> static NestedArray reshape_like(Array<T> flat,const NestedArray<S,f>& other) {
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

  bool empty() const {
    return !size();
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

  NestedArray<T> freeze() const {
    return NestedArray<T>(offsets,flat);
  }

  template<class TA> void append(const TA& other) {
    static_assert(!frozen,"To use append, set frozen=false and eventually call freeze()");
    offsets.append(offsets.back()+other.size());
    flat.extend(other);
  }

  template<class S,bool f> void extend(const NestedArray<S,f>& other) {
    static_assert(!frozen,"To use extend, set frozen=false and eventually call freeze()");
    offsets.extend(offsets.back()+other.offsets.slice(1,other.offsets.size()));
    flat.extend(other.flat);
  }
};

template<class T> inline ostream& operator<<(ostream& output, const NestedArray<T>& a) {
  output << '[';
  for (int i=0;i<a.size();i++) {
    if (i)
      output << ',';
    output << a[i];
  }
  return output<<']';
}

template<class TA> NestedArray<typename TA::value_type> make_nested(const TA& a0) {
  NestedArray<typename TA::value_type> nested(asarray(vec((int)a0.size())),false);
  nested[0] = a0;
  return nested;
}

template<class TA> NestedArray<typename TA::value_type> make_nested(const TA& a0, const TA& a1) {
  NestedArray<typename TA::value_type> nested(asarray(vec((int)a0.size(),(int)a1.size())),false);
  nested[0] = a0;
  nested[1] = a1;
  return nested;
}

template<class TA> NestedArray<typename TA::value_type::value_type,false> asnested(const TA& a) {
  return NestedArray<typename TA::value_type::value_type,false>::copy(a);
}

template<class T,bool frozen> const NestedArray<T,frozen>& asnested(const NestedArray<T,frozen>& a) {
  return a;
}

#ifdef OTHER_PYTHON
OTHER_CORE_EXPORT PyObject* nested_array_to_python_helper(PyObject* offsets, PyObject* flat);
OTHER_CORE_EXPORT Vector<Ref<>,2> nested_array_from_python_helper(PyObject* object);

template<class T> PyObject* to_python(const NestedArray<T>& array) {
  if (PyObject* offsets = to_python(array.offsets)) {
    if (PyObject* flat = to_python(array.flat))
      return nested_array_to_python_helper(offsets,flat);
    else
      Py_DECREF(offsets);
  }
  return 0;
}

template<class T> struct FromPython<NestedArray<T>>{static NestedArray<T> convert(PyObject* object);};
template<class T> NestedArray<T> FromPython<NestedArray<T>>::convert(PyObject* object) {
  const auto fields = nested_array_from_python_helper(object);
  NestedArray<T> self;
  self.offsets = from_python<Array<const int>>(fields.x);
  self.flat = from_python<Array<T>>(fields.y);
  return self;
}
#endif

}
