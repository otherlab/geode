#pragma once

/*

  This file contains numpy definitions that are necessary for some python conversions,
  but which are safe to include even if you don't want to depend on numpy.

*/



#include <geode/python/config.h>
#include <geode/utility/config.h>
#include <geode/array/Array.h>

namespace geode {
// Use an unnamed namespace since a given instantiation of these functions should appear in only one object file
namespace {

#ifndef GEODE_PYTHON

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

#else

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

#endif

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

}
}
