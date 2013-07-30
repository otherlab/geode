//#####################################################################
// Numpy interface functions
//#####################################################################
#include <other/core/python/numpy.h>
#include <other/core/python/wrap.h>
#include <boost/detail/endian.hpp>
#include <boost/cstdint.hpp>
#include <stdio.h>
#ifdef OTHER_PYTHON
#include <numpy/npy_common.h>
#endif
namespace other {

#ifdef OTHER_PYTHON

#ifndef NPY_ARRAY_ALIGNED
#define NPY_ARRAY_ALIGNED NPY_ALIGNED
#endif

void throw_dimension_mismatch() {
  PyErr_SetString(PyExc_ValueError,"dimension mismatch");
  throw PythonError();
}

void throw_not_owned() {
  PyErr_SetString(PyExc_ValueError,"arrays which don't own their data can't be converted to python");
  throw PythonError();
}

bool is_numpy_array(PyObject* o) {
  return PyArray_Check(o);
}

PyArray_Descr* numpy_descr_from_type(int type_num) {
  return PyArray_DescrFromType(type_num);
}

PyObject* numpy_from_any(PyObject* op, PyArray_Descr* dtype, int min_depth, int max_depth, int requirements, PyObject* context) {
  if (op==Py_None) // PyArray_FromAny silently converts None to a singleton nan, which is not cool
    throw TypeError("expected numpy array, got None");
  return PyArray_FromAny(op,dtype,min_depth,max_depth,requirements,context);
}

PyObject* numpy_new_from_descr(PyTypeObject* subtype, PyArray_Descr* descr, int nd, npy_intp* dims, npy_intp* strides, void* data, int flags, PyObject* obj) {
  return PyArray_NewFromDescr(subtype,descr,nd,dims,strides,data,flags,obj);
}

PyTypeObject* numpy_array_type() {
  return &PyArray_Type;
}

static PyTypeObject* recarray = 0;
PyTypeObject* numpy_recarray_type() {
  OTHER_ASSERT(recarray);
  return recarray;
}
static void _set_recarray_type(PyObject* type) {
  OTHER_ASSERT(PyType_Check(type));
  Py_INCREF(type);
  recarray = (PyTypeObject*)type;
}

void throw_array_conversion_error(PyObject* object, int flags, int rank_range, PyArray_Descr* descr) {
  if (!PyArray_Check(object))
    PyErr_Format(PyExc_TypeError, "expected numpy array, got %s", object->ob_type->tp_name);
  else if (!PyArray_EquivTypes(PyArray_DESCR((PyArrayObject*)object), descr))
    PyErr_Format(PyExc_TypeError, "expected array type %s, got %s", descr->typeobj->tp_name, PyArray_DESCR((PyArrayObject*)object)->typeobj->tp_name);
  else if (!PyArray_CHKFLAGS((PyArrayObject*)object,flags)){
    if (flags&NPY_ARRAY_WRITEABLE && !PyArray_ISWRITEABLE((PyArrayObject*)object))
      PyErr_SetString(PyExc_TypeError, "unwriteable array");
    else if (flags&NPY_ARRAY_ALIGNED && !PyArray_ISALIGNED((PyArrayObject*)object))
      PyErr_SetString(PyExc_TypeError, "unaligned array");
    else if (flags&NPY_ARRAY_C_CONTIGUOUS && !PyArray_ISCONTIGUOUS((PyArrayObject*)object))
      PyErr_SetString(PyExc_TypeError, "noncontiguous array");
    else
      PyErr_SetString(PyExc_TypeError, "unknown array flag mismatch");}
  else if (rank_range>=0)
    PyErr_Format(PyExc_ValueError, "expected rank %d, got %d", rank_range, PyArray_NDIM((PyArrayObject*)object));
  else
    PyErr_Format(PyExc_ValueError, "expected rank at least %d, got %d", -rank_range-1, PyArray_NDIM((PyArrayObject*)object));
  throw PythonError();
}

void check_numpy_conversion(PyObject* object, int flags, int rank_range, PyArray_Descr* descr) {
  const int rank = PyArray_NDIM((PyArrayObject*)object);
  const int min_rank = rank_range<0?-rank_range-1:rank_range,max_rank=rank_range<0?100:rank_range;
  if (!PyArray_CHKFLAGS((PyArrayObject*)object,flags) || min_rank>rank || rank>max_rank || !PyArray_EquivTypes(PyArray_DESCR((PyArrayObject*)object),descr))
    throw_array_conversion_error(object,flags,rank_range,descr);
}

#else // !defined(OTHER_PYTHON)

// CHAR_BIT isn't defined for some build configurations so we use __CHAR_BIT__ instead which seems to work in both clang and gcc
// It would probably be safe to just use 8 if this fails
#define OTHER_CHAR_BIT __CHAR_BIT__

// Modified from numpy/npy_common.h
typedef unsigned char npy_bool;
#define NPY_BITSOF_BOOL (sizeof(npy_bool)*OTHER_CHAR_BIT)
#define NPY_BITSOF_CHAR (sizeof(char)*OTHER_CHAR_BIT)
#define NPY_BITSOF_SHORT (sizeof(short)*OTHER_CHAR_BIT)
#define NPY_BITSOF_INT (sizeof(int)*OTHER_CHAR_BIT)
#define NPY_BITSOF_LONG (sizeof(long)*OTHER_CHAR_BIT)
#define NPY_BITSOF_LONGLONG (sizeof(long long)*OTHER_CHAR_BIT)
#define NPY_BITSOF_FLOAT (sizeof(float)*OTHER_CHAR_BIT)
#define NPY_BITSOF_DOUBLE (sizeof(double)*OTHER_CHAR_BIT)
#define NPY_BITSOF_LONGDOUBLE (sizeof(long double)*OTHER_CHAR_BIT)

// Lifted from numpy/ndarraytypes.h
enum NPY_TYPECHAR { NPY_GENBOOLLTR ='b',
                    NPY_SIGNEDLTR = 'i',
                    NPY_UNSIGNEDLTR = 'u',
                    NPY_FLOATINGLTR = 'f',
                    NPY_COMPLEXLTR = 'c'
};

#endif

size_t fill_numpy_header(Array<uint8_t>& header,int rank,const npy_intp* dimensions,int type_num) {
  // Get dtype info
  int bits;
  char letter;
  switch (type_num) {
    #define CASE(T,K) case NPY_##T:bits=NPY_BITSOF_##T;letter=NPY_##K##LTR;break;
    #ifndef NPY_BITSOF_BYTE
    #define NPY_BITSOF_BYTE 8
    #endif
    #define NPY_BITSOF_UBYTE 8
    #define NPY_BITSOF_USHORT NPY_BITSOF_SHORT
    #define NPY_BITSOF_UINT NPY_BITSOF_INT
    #define NPY_BITSOF_ULONG NPY_BITSOF_LONG
    #define NPY_BITSOF_ULONGLONG NPY_BITSOF_LONGLONG
    CASE(BOOL,GENBOOL)
    CASE(BYTE,SIGNED)
    CASE(UBYTE,UNSIGNED)
    CASE(SHORT,SIGNED)
    CASE(USHORT,UNSIGNED)
    CASE(INT,SIGNED)
    CASE(UINT,UNSIGNED)
    CASE(LONG,SIGNED)
    CASE(ULONG,UNSIGNED)
    CASE(LONGLONG,SIGNED)
    CASE(ULONGLONG,UNSIGNED)
    CASE(FLOAT,FLOATING)
    CASE(DOUBLE,FLOATING)
    CASE(LONGDOUBLE,FLOATING)
    #undef CASE
    default: throw ValueError("Unknown dtype");
  }
  const int bytes = bits/8;

  // Endianness
#if defined(BOOST_LITTLE_ENDIAN)
  const char endian = '<';
#elif defined(BOOST_BIG_ENDIAN)
  const char endian = '>';
#else
#error "Unknown endianness"
#endif

  // Construct header
  const char magic_version[8] = {(char)0x93,'N','U','M','P','Y',1,0};
  header.resize(256,false,false);
  char* const base = (char*)header.data();
  memcpy(base,magic_version,8);
  int len = 10;
  len += sprintf(base+len,"{'descr': '%c%c%d', 'fortran_order': False, 'shape': (",endian,letter,bytes);
  size_t total_size = 1;
  for (int i=0;i<rank;i++) {
    total_size *= dimensions[i];
    len += sprintf(base+len,"%ld%s",dimensions[i],rank==1||i<rank-1?",":"");
  }
  strcpy(base+len,"), }");
  len+=4;
  while ((len+1)&15)
    header[len++] = ' ';
  OTHER_ASSERT(((len+1)&15)==0);
  header[len++] = '\n';
  OTHER_ASSERT((len&15)==0);
  uint16_t header_len = uint16_t(len-10);
  OTHER_ASSERT(header_len==len-10);
#ifdef BOOST_BIG_ENDIAN
  // Switch header_len to little endian
  swap(((char*)&header_len)[0],((char*)&header_len)[1]);
#endif
  memcpy(base+8,&header_len,2);
  header.resize(len);
  return bytes*total_size;
}

void write_numpy(const string& filename,int rank,const npy_intp* dimensions,int type_num,void* data) {
  // Fill header
  Array<uint8_t> header;
  size_t data_size = fill_numpy_header(header,rank,dimensions,type_num);

  // Write npy file
  FILE* file = fopen(filename.c_str(),"wb");
  if(!file) throw OSError("Can't open "+filename+" for writing");
  fwrite(header.data(),1,header.size(),file);
  fwrite(data,1,data_size,file);
  fclose(file);
}

}
using namespace other;

void wrap_numpy() {
#ifdef OTHER_PYTHON
  OTHER_FUNCTION(_set_recarray_type)
#endif
}
