//#####################################################################
// Python conversions for Array and NdArray
//#####################################################################
//
// Include this header if you need to register array conversion for new types.
//
//#####################################################################
#include <geode/array/Array.h>
#include <geode/array/Array2d.h>
#include <geode/array/Array3d.h>
#include <geode/array/Array4d.h>
#include <geode/array/NdArray.h>
#include <geode/array/Nested.h>
#include <geode/python/numpy.h>
#include <geode/python/stl.h>
namespace geode {

#ifdef GEODE_PYTHON

template<class T,int d> PyObject* to_python(const Array<T,d>& array) {
  return to_numpy(array);
}

template<class T,int d> Array<T,d> FromPython<Array<T,d>>::convert(PyObject* object) {
  return from_numpy<Array<T,d>>(object);
}

#define ARRAY_CONVERSIONS_HELPER(d,...) \
  template GEODE_CORE_EXPORT PyObject* to_python<__VA_ARGS__,d>(const Array<__VA_ARGS__,d>&); \
  template GEODE_CORE_EXPORT Array<__VA_ARGS__,d> FromPython<Array<__VA_ARGS__,d>>::convert(PyObject*);
#define ARRAY_CONVERSIONS(d,...) \
  ARRAY_CONVERSIONS_HELPER(d,__VA_ARGS__) \
  ARRAY_CONVERSIONS_HELPER(d,const __VA_ARGS__)

template<class T> PyObject* to_python(const NdArray<T>& array) {
  return to_numpy(array);
}

template<class T> NdArray<T> FromPython<NdArray<T>>::convert(PyObject* object) {
  return from_numpy<NdArray<T>>(object);
}

#define NDARRAY_CONVERSIONS_HELPER(...) \
  template GEODE_CORE_EXPORT PyObject* to_python<__VA_ARGS__>(const NdArray<__VA_ARGS__>&); \
  template GEODE_CORE_EXPORT NdArray<__VA_ARGS__> FromPython<NdArray<__VA_ARGS__>>::convert(PyObject*);
#define NDARRAY_CONVERSIONS(...) \
  NDARRAY_CONVERSIONS_HELPER(__VA_ARGS__) \
  NDARRAY_CONVERSIONS_HELPER(const __VA_ARGS__)

template<class T> PyObject* to_python(const Nested<T>& array) {
  if (PyObject* offsets = to_python(array.offsets)) {
    if (PyObject* flat = to_python(array.flat))
      return nested_array_to_python_helper(offsets,flat);
    else
      Py_DECREF(offsets);
  }
  return 0;
}

template<class T> Nested<T> FromPython<Nested<T>>::convert(PyObject* object) {
  if (is_nested_array(object)) {
    // Already a Python Nested object, so conversion is easy
    const auto fields = nested_array_from_python_helper(object);
    Nested<T> self;
    self.offsets = from_python<Array<const int>>(fields.x);
    self.flat = from_python<Array<T>>(fields.y);
    return self;
  } else if (is_numpy_array(object)) {
    // 2D numpy arrays can be handled more quickly than general sequences of sequences
    const auto data = from_python<Array<T,2>>(object);
    const auto offsets = data.n*arange(data.m+1);
    return Nested<T>(offsets.copy(),data.flat);
  } else {
    // Convert via an array of arrays
    return Nested<T>::copy(from_python<vector<Array<T>>>(object));
  }
}

#define NESTED_CONVERSIONS_HELPER(...) \
  template GEODE_CORE_EXPORT PyObject* to_python<__VA_ARGS__>(const Nested<__VA_ARGS__>&); \
  template GEODE_CORE_EXPORT Nested<__VA_ARGS__> FromPython<Nested<__VA_ARGS__>>::convert(PyObject*);
#define NESTED_CONVERSIONS(...) \
  NESTED_CONVERSIONS_HELPER(__VA_ARGS__) \
  NESTED_CONVERSIONS_HELPER(const __VA_ARGS__)

#else // non-python stubs

#define ARRAY_CONVERSIONS(...)
#define NDARRAY_CONVERSIONS(...)
#define NESTED_CONVERSIONS(...)

#endif

}
