//#####################################################################
// Numpy interface functions
//#####################################################################
#pragma once

#define PY_ARRAY_UNIQUE_SYMBOL _try_python_array_api
#define NPY_NO_DEPRECATED_API

#ifndef OTHER_IMPORT_NUMPY
#define NO_IMPORT_ARRAY
#endif

#include <other/core/python/config.h>
#include <numpy/arrayobject.h>

#include <other/core/array/Array.h>
#include <other/core/array/IndirectArray.h>
#include <other/core/python/exceptions.h>
#include <other/core/utility/config.h>
#include <other/core/utility/const_cast.h>
namespace other {

typedef Py_intptr_t npy_intp;

OTHER_EXPORT void OTHER_NORETURN(throw_dimension_mismatch());
OTHER_EXPORT void OTHER_NORETURN(throw_not_owned());
OTHER_EXPORT void OTHER_NORETURN(throw_array_conversion_error(PyObject* object,int flags,int rank_range,PyArray_Descr* descr));
OTHER_EXPORT size_t fill_numpy_header(Array<uint8_t>& header,int rank,const npy_intp* dimensions,int type_num); // Returns total data size in bytes
OTHER_EXPORT void write_numpy(const string& filename,int rank,const npy_intp* dimensions,int type_num,void* data);

// Stay compatible with old versions of numpy
#ifndef NPY_ARRAY_WRITEABLE
#define NPY_ARRAY_WRITEABLE NPY_WRITEABLE
#define NPY_ARRAY_FORCECAST NPY_FORCECAST
#define NPY_ARRAY_CARRAY_RO NPY_CARRAY_RO
#define NPY_ARRAY_CARRAY NPY_CARRAY
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#endif

// The numpy API changed incompatibly without incrementing the NPY_VERSION define.  Therefore, we use the
// fact that PyArray_BASE switched from being a macro in the old version to an inline function in the new.
#if defined(PyArray_BASE) && !defined(PyArray_CLEARFLAGS)
static inline void PyArray_CLEARFLAGS(PyArrayObject* array,int flags) {
  PyArray_FLAGS(array) &= ~flags;
}
static inline void PyArray_ENABLEFLAGS(PyArrayObject* array,int flags) {
  PyArray_FLAGS(array) |= flags;
}
static inline void PyArray_SetBaseObject(PyArrayObject* array,PyObject* base) {
  PyArray_BASE(array) = base;
}
#endif

// Use an unnamed namespace since a given instantiation of these functions should appear in only one object file
namespace {

// NumpyIsScalar
template<class T> struct NumpyIsScalar:public mpl::false_{};
template<class T> struct NumpyIsScalar<const T>:public NumpyIsScalar<T>{};
template<> struct NumpyIsScalar<bool>:public mpl::true_{};
template<> struct NumpyIsScalar<char>:public mpl::true_{};
template<> struct NumpyIsScalar<unsigned char>:public mpl::true_{};
template<> struct NumpyIsScalar<short>:public mpl::true_{};
template<> struct NumpyIsScalar<unsigned short>:public mpl::true_{};
template<> struct NumpyIsScalar<int>:public mpl::true_{};
template<> struct NumpyIsScalar<unsigned int>:public mpl::true_{};
template<> struct NumpyIsScalar<long>:public mpl::true_{};
template<> struct NumpyIsScalar<unsigned long>:public mpl::true_{};
template<> struct NumpyIsScalar<long long>:public mpl::true_{};
template<> struct NumpyIsScalar<unsigned long long>:public mpl::true_{};
template<> struct NumpyIsScalar<float>:public mpl::true_{};
template<> struct NumpyIsScalar<double>:public mpl::true_{};
template<> struct NumpyIsScalar<long double>:public mpl::true_{};

// NumpyIsStatic
template<class T> struct NumpyIsStatic:public NumpyIsScalar<T>{};
template<class T,int d> struct NumpyIsStatic<Vector<T,d> >:public mpl::true_{};
template<class T,int m,int n> struct NumpyIsStatic<Matrix<T,m,n> >:public mpl::bool_<(m>=1 && n>=1)>{};

// NumpyScalar: Recursively extract type information from array types
template<class T> struct NumpyScalar; // map from primitive types to numpy type ids
template<class T> struct NumpyScalar<const T>:public NumpyScalar<T>{};

template<> struct NumpyScalar<bool>{enum {value=NPY_BOOL};};
template<> struct NumpyScalar<char>{enum {value=NPY_BYTE};};
template<> struct NumpyScalar<unsigned char>{enum {value=NPY_UBYTE};};
template<> struct NumpyScalar<short>{enum {value=NPY_SHORT};};
template<> struct NumpyScalar<unsigned short>{enum {value=NPY_USHORT};};
template<> struct NumpyScalar<int>{enum {value=NPY_INT};};
template<> struct NumpyScalar<unsigned int>{enum {value=NPY_UINT};};
template<> struct NumpyScalar<long>{enum {value=NPY_LONG};};
template<> struct NumpyScalar<unsigned long>{enum {value=NPY_ULONG};};
template<> struct NumpyScalar<long long>{enum {value=NPY_LONGLONG};};
template<> struct NumpyScalar<unsigned long long>{enum {value=NPY_ULONGLONG};};
template<> struct NumpyScalar<float>{enum {value=NPY_FLOAT};};
template<> struct NumpyScalar<double>{enum {value=NPY_DOUBLE};};
template<> struct NumpyScalar<long double>{enum {value=NPY_LONGDOUBLE};};

template<class T,int d> struct NumpyScalar<Vector<T,d> >:public NumpyScalar<T>{};
template<class T,int m,int n> struct NumpyScalar<Matrix<T,m,n> >:public NumpyScalar<T>{};
template<class T,int d> struct NumpyScalar<Array<T,d> >:public NumpyScalar<T>{};
template<class T,int d> struct NumpyScalar<RawArray<T,d> >:public NumpyScalar<T>{};
template<class T> struct NumpyScalar<NdArray<T> >:public NumpyScalar<T>{};

// NumpyDescr
template<class T> struct NumpyDescr{static PyArray_Descr* descr(){return PyArray_DescrFromType(NumpyScalar<T>::value);}};
template<class T> struct NumpyDescr<const T>:public NumpyDescr<T>{};
template<class T,int d> struct NumpyDescr<Vector<T,d> >:public NumpyDescr<T>{};
template<class T,int m,int n> struct NumpyDescr<Matrix<T,m,n> >:public NumpyDescr<T>{};
template<class T,int d> struct NumpyDescr<Array<T,d> >:public NumpyDescr<T>{};
template<class T,int d> struct NumpyDescr<RawArray<T,d> >:public NumpyDescr<T>{};
template<class T> struct NumpyDescr<NdArray<T> >:public NumpyDescr<T>{};

// NumpyArrayType
template<class T> struct NumpyArrayType{static PyTypeObject* type(){return &PyArray_Type;}};
template<class T> struct NumpyArrayType<const T>:public NumpyArrayType<T>{};
template<class T,int d> struct NumpyArrayType<Vector<T,d> >:public NumpyArrayType<T>{};
template<class T,int d> struct NumpyArrayType<Array<T,d> >:public NumpyArrayType<T>{};
template<class T,int d> struct NumpyArrayType<RawArray<T,d> >:public NumpyArrayType<T>{};
template<class T> struct NumpyArrayType<NdArray<T> >:public NumpyArrayType<T>{};

// Struct NumpyMinRank/NumpyMaxRank: Extract rank of an array type
template<class T,class Enable=void> struct NumpyRank;
template<class T> struct NumpyRank<T,typename boost::enable_if<mpl::and_<NumpyIsScalar<T>,mpl::not_<boost::is_const<T>>>>::type>:public mpl::int_<0>{};
template<class T> struct NumpyRank<const T>:public NumpyRank<T>{};

template<class T,int d> struct NumpyRank<Vector<T,d> >:public mpl::int_<1+NumpyRank<T>::value>{};
template<class T,int m,int n> struct NumpyRank<Matrix<T,m,n> >:public mpl::int_<2+NumpyRank<T>::value>{};
template<class T,int d> struct NumpyRank<Array<T,d> >:public mpl::int_<d+NumpyRank<T>::value>{};
template<class T,int d> struct NumpyRank<RawArray<T,d> >:public mpl::int_<d+NumpyRank<T>::value>{};
template<class T> struct NumpyRank<NdArray<T> >:public mpl::int_<-1-NumpyRank<T>::value>{}; // -r-1 means r or higher

// Extract possibly runtime-variable rank
template<class T> int numpy_rank(const T&) {
  return NumpyRank<T>::value;
}
template<class T> int numpy_rank(const NdArray<T>& array) {
  return array.rank()+NumpyRank<T>::value;
}

// Struct NumpyInfo: Recursively shape information from statically sized types

template<class T> struct NumpyInfo { static void dimensions(npy_intp* dimensions) {
  BOOST_STATIC_ASSERT(NumpyRank<T>::value==0);
}};

template<class T> struct NumpyInfo<const T> : public NumpyInfo<T>{};

template<class T,int d> struct NumpyInfo<Vector<T,d> > { static void dimensions(npy_intp* dimensions) {
  dimensions[0] = d;
  NumpyInfo<T>::dimensions(dimensions+1);
}};

template<class T,int m,int n> struct NumpyInfo<Matrix<T,m,n> > { static void dimensions(npy_intp* dimensions) {
  dimensions[0] = m;
  dimensions[1] = n;
  NumpyInfo<T>::dimensions(dimensions+2);
}};

// Function Numpy_Info: Recursively extract type and shape information from dynamically sized types

template<class TV> typename boost::enable_if<NumpyIsStatic<TV> >::type
numpy_info(const TV& block, void*& data, npy_intp* dimensions) {
  data = const_cast_(&block);
  NumpyInfo<TV>::dimensions(dimensions);
}

template<class T,int d> void
numpy_info(const Array<T,d>& array, void*& data, npy_intp* dimensions) {
  data = (void*)array.data();
  const Vector<npy_intp,d> sizes(array.sizes());
  for (int i=0;i<d;i++) dimensions[i] = sizes[i];
  NumpyInfo<T>::dimensions(dimensions+d);
}

template<class T,int d> void
numpy_info(const RawArray<T,d>& array, void*& data, npy_intp* dimensions) {
  data = (void*)array.data();
  const Vector<npy_intp,d> sizes(array.sizes());
  for (int i=0;i<d;i++) dimensions[i] = sizes[i];
  NumpyInfo<T>::dimensions(dimensions+d);
}

template<class T> void
numpy_info(const NdArray<T>& array, void*& data, npy_intp* dimensions) {
  data = (void*)array.data();
  for (int i=0;i<array.rank();i++) dimensions[i] = array.shape[i];
  NumpyInfo<T>::dimensions(dimensions+array.rank());
}

// Numpy_Shape_Match: Check whether dynamic type can be resized to fit a given numpy array

template<class T> typename boost::enable_if<NumpyIsScalar<T>,bool>::type
numpy_shape_match(mpl::identity<T>,int rank,const npy_intp* dimensions) {
  return true;
}

template<class TV> typename boost::enable_if<mpl::and_<NumpyIsStatic<TV>,mpl::not_<NumpyIsScalar<TV> > >,bool>::type
numpy_shape_match(mpl::identity<TV>, int rank, const npy_intp* dimensions) {
  if (rank!=NumpyRank<TV>::value) return false;
  npy_intp subdimensions[rank];
  NumpyInfo<TV>::dimensions(subdimensions);
  for (int i=0;i<rank;i++) if(dimensions[i]!=subdimensions[i]) return false;
  return true;
}

template<class T> bool
numpy_shape_match(mpl::identity<const T>, int rank, const npy_intp* dimensions) {
  return numpy_shape_match(mpl::identity<T>(),rank,dimensions);
}

template<class T,int d> bool
numpy_shape_match(mpl::identity<Array<T,d> >, int rank, const npy_intp* dimensions) {
  return numpy_shape_match(mpl::identity<T>(),rank-d,dimensions+d);
}

template<class T,int d> bool
numpy_shape_match(mpl::identity<RawArray<T,d> >, int rank, const npy_intp* dimensions) {
  return numpy_shape_match(mpl::identity<T>(),rank-d,dimensions+d);
}

template<class T> bool
numpy_shape_match(mpl::identity<NdArray<T> >, int rank, const npy_intp* dimensions) {
  return numpy_shape_match(mpl::identity<T>(),NumpyRank<T>::value,dimensions+rank-NumpyRank<T>::value);
}

// to_numpy for static types
template<class TV> typename boost::enable_if<NumpyIsStatic<TV>,PyObject*>::type
to_numpy(const TV& x) {
  // Extract memory layout information
  const int rank = numpy_rank(x);
  void* data;
  Array<npy_intp> dimensions(rank,false);
  numpy_info(x,data,dimensions.data());

  // Make a new numpy array and copy the vector into it
  PyObject* numpy = PyArray_NewFromDescr(NumpyArrayType<TV>::type(),NumpyDescr<TV>::descr(),rank,dimensions.data(),0,0,0,0);
  if (!numpy) return 0;
  *(TV*)PyArray_DATA((PyArrayObject*)numpy) = x;

  // Mark the array nonconst so users don't expect to be changing the original
  PyArray_CLEARFLAGS((PyArrayObject*)numpy, NPY_ARRAY_WRITEABLE);
  return numpy;
}

// to_numpy for shareable array types
template<class TArray>
typename boost::enable_if<IsShareable<TArray>,PyObject*>::type
to_numpy(TArray& array) {
  // extract memory layout information
  const int rank = numpy_rank(array);
  void* data;
  Array<npy_intp> dimensions(rank,false);
  numpy_info(array,data,dimensions.data());

  // verify ownership
  PyObject* owner = array.owner();
  if (!owner && data)
    throw_not_owned();

  // wrap the existing array as a numpy array without copying data
  PyObject* numpy = PyArray_NewFromDescr(NumpyArrayType<TArray>::type(),NumpyDescr<TArray>::descr(),rank,dimensions.data(),0,data,NPY_ARRAY_CARRAY,0);
  if (!numpy) return 0;
  PyArray_ENABLEFLAGS((PyArrayObject*)numpy,NPY_ARRAY_C_CONTIGUOUS);
  if (TArray::is_const) PyArray_CLEARFLAGS((PyArrayObject*)numpy, NPY_ARRAY_WRITEABLE);

  // let numpy array share ownership with array
  if (owner)
    PyArray_SetBaseObject((PyArrayObject*)numpy, owner);
  return numpy;
}

// from_numpy for static types
template<class TV> typename boost::enable_if<NumpyIsStatic<TV>,TV>::type
from_numpy(PyObject* object) { // Borrows reference to object
  // allow conversion from 0 to static vector/matrix types
  if(PyInt_Check(object) && !PyInt_AS_LONG(object))
    return TV();

  // convert object to an array with the correct type and rank
  static const int rank = NumpyRank<TV>::value;
  PyObject* array = PyArray_FromAny(object,NumpyDescr<TV>::descr(),rank,rank,NPY_ARRAY_CARRAY_RO|NPY_ARRAY_FORCECAST,0);
  if (!array) throw_python_error();

  // ensure appropriate dimensions
  if (!numpy_shape_match(mpl::identity<TV>(),rank,PyArray_DIMS((PyArrayObject*)array))) {
    Py_DECREF(array);
    throw_dimension_mismatch();
  }

  TV result = *(const TV*)(PyArray_DATA((PyArrayObject*)array));
  Py_DECREF(array);
  return result;
}

// Build an Array<T,d> from a compatible numpy array
template<class T,int d> inline Array<T,d>
from_numpy_helper(mpl::identity<Array<T,d> >, PyObject* array) {
  PyObject* base = PyArray_BASE((PyArrayObject*)array);
  Vector<int,d> counts;
  for (int i=0;i<d;i++){
    counts[i] = (int)PyArray_DIMS((PyArrayObject*)array)[i];
    OTHER_ASSERT(counts[i]==PyArray_DIMS((PyArrayObject*)array)[i]);}
  return Array<T,d>(counts,(T*)PyArray_DATA((PyArrayObject*)array),base?base:array);
}

// Build an NdArray<T,d> from a compatible numpy array
template<class T> inline NdArray<T>
from_numpy_helper(mpl::identity<NdArray<T> >,PyObject* array) {
  PyObject* base = PyArray_BASE((PyArrayObject*)array);
  Array<int> shape(PyArray_NDIM((PyArrayObject*)array)-NumpyRank<T>::value,false);
  for (int i=0;i<shape.size();i++){
    shape[i] = (int)PyArray_DIMS((PyArrayObject*)array)[i];
    OTHER_ASSERT(shape[i]==PyArray_DIMS((PyArrayObject*)array)[i]);}
  return NdArray<T>(shape,(T*)PyArray_DATA((PyArrayObject*)array),base?base:array);
}

// from_numpy for shareable arrays
template<class TArray> typename boost::enable_if<IsShareable<TArray>,TArray>::type
from_numpy(PyObject* object) { // borrows reference to object
  const int flags = TArray::is_const?NPY_ARRAY_CARRAY_RO:NPY_ARRAY_CARRAY;
  const int rank_range = NumpyRank<TArray>::value;
  PyArray_Descr* const descr = NumpyDescr<TArray>::descr();
  const int min_rank = rank_range<0?-rank_range-1:rank_range,max_rank=rank_range<0?100:rank_range;

  if (PyArray_Check(object)) {
    // Already a numpy array: require an exact match to avoid hidden performance issues
    int rank = PyArray_NDIM((PyArrayObject*)object);
    if (!PyArray_CHKFLAGS((PyArrayObject*)object,flags) || min_rank>rank || rank>max_rank || !PyArray_EquivTypes(PyArray_DESCR((PyArrayObject*)object), descr))
      throw_array_conversion_error(object,flags,rank_range,descr);
    if (!numpy_shape_match(mpl::identity<TArray>(),rank,PyArray_DIMS((PyArrayObject*)object)))
      throw_dimension_mismatch();
    return from_numpy_helper(mpl::identity<TArray>(),object);
  } else if (!TArray::is_const)
    throw_type_error(object, &PyArray_Type);

  // if we're converting to a const array, and the input isn't already numpy, allow any matching nested sequence
  PyObject* array = PyArray_FromAny(object,descr,min_rank,max_rank,flags,0);
  if(!array) throw_python_error();

  // ensure appropriate dimension
  int rank = PyArray_NDIM((PyArrayObject*)array);
  if (!numpy_shape_match(mpl::identity<TArray>(),rank,PyArray_DIMS((PyArrayObject*)array))) {
    Py_DECREF(array);
    throw_dimension_mismatch();
  }

  TArray result = from_numpy_helper(mpl::identity<TArray>(),array);
  Py_DECREF(array);
  return result;
}

// Write a numpy-convertible array to an .npy file
// Note: Unlike other functions in this file, it is safe to call write_numpy without initializing either Python or Numpy.
template<class TArray> void
write_numpy(const string& filename, const TArray& array) {
  // Extract memory layout information
  const int rank = numpy_rank(array);
  void* data;
  Array<npy_intp> dimensions(rank,false);
  numpy_info(array,data,dimensions.data());
  // Write
  other::write_numpy(filename,rank,dimensions.data(),NumpyScalar<TArray>::value,data);
}

// Generate an .npy file header for a numpy-convertible array
// Note: Unlike other functions in this file, it is safe to call fill_numpy_header without initializing either Python or Numpy.
template<class TArray> size_t
fill_numpy_header(Array<uint8_t>& header,const TArray& array) {
  // Extract memory layout information
  const int rank = numpy_rank(array);
  void* data;
  Array<npy_intp> dimensions(rank,false);
  numpy_info(array,data,dimensions.data());
  // Fill header
  return other::fill_numpy_header(header,rank,dimensions.data(),NumpyScalar<TArray>::value);
}

}
}
