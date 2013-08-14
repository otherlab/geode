//#####################################################################
// Class Nested
//#####################################################################
//
// Nested<T> is essentially Array<Array<T>>, but stored in flat form for efficiency.
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
class Nested {
  struct Unusable {};
public:
  typedef typename Array<T>::Element Element;
  template<class S,bool f> struct Compatible { static const bool value = boost::is_same<Element,typename boost::remove_const<S>::type>::value && boost::is_const<T>::value>=boost::is_const<S>::value; };

  // When growing an array incrementally via append or extend, set frozen=false to make offsets mutable.
  typedef Array<typename mpl::if_c<frozen,const int,int>::type> Offsets;

  Offsets offsets;
  Array<T> flat;

  Nested()
    : offsets(nested_array_offsets(Array<const int>())) {}

  Nested(RawArray<const int> lengths, bool initialize=true)
    : offsets(nested_array_offsets(lengths)), flat(offsets.back(),initialize) {}

  Nested(const Nested& other)
    : offsets(other.offsets)
    , flat(other.flat) {}

  template<class S,bool f> Nested(const Nested<S,f>& other, typename boost::enable_if<Compatible<S,f>,Unusable>::type unusable=Unusable())
    : offsets(other.offsets)
    , flat(other.flat) {}

  // Build from manually constructed offsets and flat arrays
  Nested(const Offsets& offsets, const Array<T>& flat)
    : offsets(offsets), flat(flat) {
    assert(offsets.back()==flat.size());
  }

  // Note: To convert vector<vector<T>> to a Nested, use copy below

  template<class S,bool f> Nested& operator=(const Nested<S,f>& other) {
    offsets = other.offsets;
    flat = other.flat;
    return *this;
  }

  template<class S,bool f> static Nested zeros_like(const Nested<S,f>& other) {
    Nested array;
    array.offsets = other.offsets;
    array.flat.resize(array.offsets.back());
    return array;
  }

  template<class S,bool f> static Nested empty_like(const Nested<S,f>& other) {
    Nested array;
    array.offsets = other.offsets;
    array.flat.resize(array.offsets.back(),false,false);
    return array;
  }

  template<class S,bool f> static Nested reshape_like(Array<T> flat,const Nested<S,f>& other) {
    OTHER_ASSERT(other.flat.size()==flat.size());
    Nested array;
    array.offsets = other.offsets;
    array.flat = flat;
    return array;
  }

  template<class TA> static Nested copy(const TA& other) {
    const int n = (int)other.size();
    Array<int> offsets(n+1,false);
    offsets[0] = 0;
    for (int i=0;i<n;i++)
      offsets[i+1] = offsets[i]+(int)other[i].size();
    Array<Element> flat(offsets[n],false);
    for (int i=0;i<n;i++)
      flat.slice(offsets[i],offsets[i+1]) = other[i];
    Nested self;
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

  bool operator==(const Nested& v) const {
    return (offsets == v.offsets) && (flat == v.flat);
  }

  Nested<Element> copy() const {
    Nested<Element> copy;
    copy.offsets = offsets;
    copy.flat = flat.copy();
    return copy;
  }

  ArrayIter<Nested> begin() const {
    return ArrayIter<Nested>(*this,0);
  }

  ArrayIter<Nested> end() const {
    return ArrayIter<Nested>(*this,size());
  }

  RawArray<T> front() const {
    return (*this)[0];
  }

  RawArray<T> back() const {
    return (*this)[size()-1];
  }

  Nested<T> freeze() const {
    return Nested<T>(offsets,flat);
  }

  template<class TA> void append(const TA& other) {
    static_assert(!frozen,"To use append, set frozen=false and eventually call freeze()");
    offsets.append(offsets.back()+other.size());
    flat.extend(other);
  }

  template<class S,bool f> void extend(const Nested<S,f>& other) {
    static_assert(!frozen,"To use extend, set frozen=false and eventually call freeze()");
    offsets.extend(offsets.back()+other.offsets.slice(1,other.offsets.size()));
    flat.extend(other.flat);
  }

  // Add an empty array to the back
  void append_empty() {
    static_assert(!frozen,"To use append_empty, set frozen=false and eventually call freeze()");
    offsets.append(offsets.back());
  }

  // Append a single element to the last subarray
  void append_to_back(const T& element) {
    static_assert(!frozen,"To use append_to_back, set frozen=false and eventually call freeze()");
    assert(!empty());
    offsets.back() += 1;
    flat.append(element);
  }

  // Extend the last subarray
  template<class TArray> void extend_back(const TArray& append_array) {
    static_assert(!frozen,"To use extend_back, set frozen=false and eventually call freeze()");
    STATIC_ASSERT_SAME(typename boost::remove_const<T>::type,typename boost::remove_const<typename TArray::value_type>::type);
    assert(!empty());
    flat.extend(append_array);
    offsets.back() += append_array.size();
  }
};

template<class T,bool f> inline ostream& operator<<(ostream& output, const Nested<T,f>& a) {
  output << '[';
  for (int i=0;i<a.size();i++) {
    if (i)
      output << ',';
    output << a[i];
  }
  return output<<']';
}

template<class TA> Nested<typename TA::value_type> make_nested(const TA& a0) {
  Nested<typename TA::value_type> nested(asarray(vec((int)a0.size())),false);
  nested[0] = a0;
  return nested;
}

template<class TA> Nested<typename TA::value_type> make_nested(const TA& a0, const TA& a1) {
  Nested<typename TA::value_type> nested(asarray(vec((int)a0.size(),(int)a1.size())),false);
  nested[0] = a0;
  nested[1] = a1;
  return nested;
}

template<class T> std::vector<std::vector<typename boost::remove_const<T>::type>> as_vectors(Nested<T> const &a) {
  std::vector<std::vector<typename boost::remove_const<T>::type>> r;
  for (auto ar : a) {
    r.push_back(std::vector<T>(ar.begin(), ar.end()));
  }
  return r;
}

// we have to put "parentheses" around TA::value_type to prevent MSVC from thinking TA::value_type::value_type is a constructor.
template<class TA> Nested<typename First<typename TA::value_type,void>::type::value_type,false> asnested(const TA& a) {
  return Nested<typename First<typename TA::value_type,void>::type::value_type,false>::copy(a);
}

template<class T,bool frozen> static inline const Nested<T,frozen>& asnested(const Nested<T,frozen>& a) {
  return a;
}

template<class T,bool f> static inline const Nested<T,f>& concatenate(const Nested<T,f>& a0) {
  return a0;
}

template<class T,bool f0,bool f1> Nested<typename boost::remove_const<T>::type,false> concatenate(const Nested<T,f0>& a0, const Nested<T,f1>& a1) {
  return Nested<typename boost::remove_const<T>::type,false>(concatenate(a0.offsets,a0.offsets.back()+a1.offsets.slice(1,a1.offsets.size())),
                                                             concatenate(a0.flat,a1.flat));
}

#ifdef OTHER_PYTHON
OTHER_CORE_EXPORT PyObject* nested_array_to_python_helper(PyObject* offsets, PyObject* flat);
OTHER_CORE_EXPORT Vector<Ref<>,2> nested_array_from_python_helper(PyObject* object);

template<class T> PyObject* to_python(const Nested<T>& array) {
  if (PyObject* offsets = to_python(array.offsets)) {
    if (PyObject* flat = to_python(array.flat))
      return nested_array_to_python_helper(offsets,flat);
    else
      Py_DECREF(offsets);
  }
  return 0;
}

template<class T> struct FromPython<Nested<T>>{static Nested<T> convert(PyObject* object);};
template<class T> Nested<T> FromPython<Nested<T>>::convert(PyObject* object) {
  const auto fields = nested_array_from_python_helper(object);
  Nested<T> self;
  self.offsets = from_python<Array<const int>>(fields.x);
  self.flat = from_python<Array<T>>(fields.y);
  return self;
}
#endif

} // namespace other
