// Numpy definitions without Numpy dependencies
//
// This file is independent of the Numpy headers so that we can (for example) write
// .npy files without depending on numpy.
#pragma once

#include <geode/utility/config.h>
#include <geode/array/Array.h>
namespace geode {

// Lifted from numpy/ndarraytypes.h
enum NPY_TYPES { NPY_BOOL=0,
                 NPY_BYTE, NPY_UBYTE,
                 NPY_SHORT, NPY_USHORT,
                 NPY_INT, NPY_UINT,
                 NPY_LONG, NPY_ULONG,
                 NPY_LONGLONG, NPY_ULONGLONG,
                 NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
                 NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
                 NPY_OBJECT=17,
                 NPY_STRING, NPY_UNICODE,
                 NPY_VOID
};

// NumpyIsScalar: Is a type a numpy scalar?
template<class T> struct NumpyIsScalar : public mpl::false_{};
template<class T> struct NumpyIsScalar<const T> : public NumpyIsScalar<T>{};
template<> struct NumpyIsScalar<bool> : public mpl::true_{};
template<> struct NumpyIsScalar<char> : public mpl::true_{};
template<> struct NumpyIsScalar<unsigned char> : public mpl::true_{};
template<> struct NumpyIsScalar<short> : public mpl::true_{};
template<> struct NumpyIsScalar<unsigned short> : public mpl::true_{};
template<> struct NumpyIsScalar<int> : public mpl::true_{};
template<> struct NumpyIsScalar<unsigned int> : public mpl::true_{};
template<> struct NumpyIsScalar<long> : public mpl::true_{};
template<> struct NumpyIsScalar<unsigned long> : public mpl::true_{};
template<> struct NumpyIsScalar<long long> : public mpl::true_{};
template<> struct NumpyIsScalar<unsigned long long> : public mpl::true_{};
template<> struct NumpyIsScalar<float> : public mpl::true_{};
template<> struct NumpyIsScalar<double> : public mpl::true_{};
template<> struct NumpyIsScalar<long double> : public mpl::true_{};

// NumpyIsStatic: Is a type fixed shape numpy compatible?
template<class T> struct NumpyIsStatic : public NumpyIsScalar<T>{};
template<class T,int d> struct NumpyIsStatic<Vector<T,d>> : public mpl::true_{};
template<class T,int m,int n> struct NumpyIsStatic<Matrix<T,m,n>> : public mpl::bool_<(m>=1 && n>=1)>{};

// NumpyScalar: Recursively extract type information from array types
template<class T> struct NumpyScalar; // map from primitive types to numpy type ids
template<class T> struct NumpyScalar<const T> : public NumpyScalar<T>{};
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
template<class T,int d> struct NumpyScalar<Vector<T,d>> : public NumpyScalar<T>{};
template<class T,int m,int n> struct NumpyScalar<Matrix<T,m,n>> : public NumpyScalar<T>{};
template<class T,int d> struct NumpyScalar<Array<T,d>> : public NumpyScalar<T>{};
template<class T,int d> struct NumpyScalar<RawArray<T,d>> : public NumpyScalar<T>{};
template<class T> struct NumpyScalar<NdArray<T>> : public NumpyScalar<T>{};

// NumpyRank: Extract the rank of a static rank type (-r-1 means r or higher)
template<class T,class Enable=void> struct NumpyRank;
template<class T> struct NumpyRank<T,typename enable_if<mpl::and_<NumpyIsScalar<T>,mpl::not_<is_const<T>>>>::type>
  : public mpl::int_<0>{};
template<class T> struct NumpyRank<const T> : public NumpyRank<T>{};
template<class T,int d> struct NumpyRank<Vector<T,d>> : public mpl::int_<1+NumpyRank<T>::value>{};
template<class T,int m,int n> struct NumpyRank<Matrix<T,m,n>> : public mpl::int_<2+NumpyRank<T>::value>{};
template<class T,int d> struct NumpyRank<Array<T,d>> : public mpl::int_<d+NumpyRank<T>::value>{};
template<class T,int d> struct NumpyRank<RawArray<T,d>> : public mpl::int_<d+NumpyRank<T>::value>{};
template<class T> struct NumpyRank<NdArray<T>> : public mpl::int_<-1-NumpyRank<T>::value>{}; // r or higher

// numpy_rank: Extract runtime variable rank
template<class T> static inline int numpy_rank(const T&) {
  return NumpyRank<T>::value;
}
template<class T> static inline int numpy_rank(const NdArray<T>& array) {
  return array.rank()+NumpyRank<T>::value;
}

// NumpyShape: Recursively extract shape from statically sized types
template<class T> struct NumpyShape { static void shape(long* shape) {
  static_assert(NumpyRank<T>::value==0,"");
}};

template<class T> struct NumpyShape<const T> : public NumpyShape<T>{};

template<class T,int d> struct NumpyShape<Vector<T,d>> { static void shape(long* shape) {
  shape[0] = d;
  NumpyShape<T>::shape(shape+1);
}};

template<class T,int m,int n> struct NumpyShape<Matrix<T,m,n>> { static void shape(long* shape) {
  shape[0] = m;
  shape[1] = n;
  NumpyShape<T>::shape(shape+2);
}};

// numpy_shape: Recursively extract shape information from dynamically sized types
template<class TV> typename enable_if<NumpyIsStatic<TV>>::type numpy_shape(const TV& block, long* shape) {
  NumpyShape<TV>::shape(shape);
}

template<class T,int d> void numpy_shape(const Array<T,d>& array, long* shape) {
  const Vector<long,d> sizes(array.sizes());
  for (int i=0;i<d;i++) shape[i] = sizes[i];
  NumpyShape<T>::shape(shape+d);
}

template<class T,int d> void numpy_shape(const RawArray<T,d>& array, long* shape) {
  const Vector<long,d> sizes(array.sizes());
  for (int i=0;i<d;i++) shape[i] = sizes[i];
  NumpyShape<T>::shape(shape+d);
}

template<class T> void numpy_shape(const NdArray<T>& array, long* shape) {
  for (int i=0;i<array.rank();i++) shape[i] = array.shape[i];
  NumpyShape<T>::shape(shape+array.rank());
}

// Helper functions for write_numpy and fill_numpy_header
GEODE_CORE_EXPORT size_t fill_numpy_header_helper(Array<uint8_t>& header, RawArray<const long> shape, const int type);
GEODE_CORE_EXPORT void write_numpy_helper(const string& filename, RawArray<const long> shape,
                                          const int type_num, const void* data);

// Write a numpy-convertible array to an .npy file.
template<class TArray> void write_numpy(const string& filename, const TArray& array) {
  Array<long> shape(numpy_rank(array),uninit);
  numpy_shape(array,shape.data());
  write_numpy_helper(filename,shape,NumpyScalar<TArray>::value,array.data());
}

// Generate an .npy file header for a numpy-convertible array.  Returns total data size in bytes.
template<class TArray> size_t fill_numpy_header(Array<uint8_t>& header, const TArray& array) {
  Array<long> shape(numpy_rank(array),uninit);
  numpy_shape(array,shape.data());
  return fill_numpy_header_helper(header,shape,NumpyScalar<TArray>::value);
}

// For testing purposes
Array<uint8_t> array_write_test(const string& filename, RawArray<const real,2> array);

}
