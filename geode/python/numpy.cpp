//#####################################################################
// Numpy interface functions
//#####################################################################
#include <geode/python/numpy.h>
#include <geode/python/wrap.h>
#include <geode/utility/endian.h>
#include <stdio.h>
#ifdef GEODE_PYTHON
#include <numpy/npy_common.h>
#endif
namespace geode {

#ifdef GEODE_PYTHON

#ifndef NPY_ARRAY_ALIGNED
#define NPY_ARRAY_ALIGNED NPY_ALIGNED
#endif

// Set in module.cpp
bool numpy_imported = false;
#define ASSERT_IMPORTED() \
  GEODE_ASSERT(numpy_imported,"Numpy not yet imported, probably caused by two different versions of libgeode")

void throw_dimension_mismatch() {
  PyErr_SetString(PyExc_ValueError,"dimension mismatch");
  throw PythonError();
}

void throw_not_owned() {
  PyErr_SetString(PyExc_ValueError,"arrays which don't own their data can't be converted to python");
  throw PythonError();
}

bool is_numpy_array(PyObject* o) {
  ASSERT_IMPORTED();
  return PyArray_Check(o);
}

PyArray_Descr* numpy_descr_from_type(int type_num) {
  ASSERT_IMPORTED();
  return PyArray_DescrFromType(type_num);
}

Ref<> numpy_from_any(PyObject* op, PyArray_Descr* dtype, int min_rank, int max_rank, int requirements) {
  if (op==Py_None) // PyArray_FromAny silently converts None to a singleton nan, which is not cool
    throw TypeError("Expected array, got None");

  // Perform the conversion
  ASSERT_IMPORTED();
  PyObject* const array = PyArray_FromAny(op,dtype,0,0,requirements,0);
  if (!array)
    throw_python_error();

  // Numpy produces uninformative error messages on rank mismatch, so we roll our own.
  const int rank = PyArray_NDIM((PyArrayObject*)array);
  if (   (min_rank && rank<min_rank)
      || (max_rank && rank>max_rank))
    throw ValueError(min_rank==max_rank ? format("Expected array with rank %d, got rank %d",min_rank,rank)
                            : !min_rank ? format("Expected array with rank <= %d, got rank %d",max_rank,rank)
                            : !max_rank ? format("Expected array with rank >= %d, got rank %d",min_rank,rank)
                            : format("Expected array with %d <= rank <= %d, got rank %d",min_rank,max_rank,rank));

  // Success!
  return steal_ref(*array);
}

PyObject* numpy_new_from_descr(PyTypeObject* subtype, PyArray_Descr* descr, int nd, npy_intp* dims, npy_intp* strides, void* data, int flags, PyObject* obj) {
  ASSERT_IMPORTED();
  return PyArray_NewFromDescr(subtype,descr,nd,dims,strides,data,flags,obj);
}

PyTypeObject* numpy_array_type() {
  ASSERT_IMPORTED();
  return &PyArray_Type;
}

static PyTypeObject* recarray = 0;
PyTypeObject* numpy_recarray_type() {
  GEODE_ASSERT(recarray);
  return recarray;
}
static void _set_recarray_type(PyObject* type) {
  GEODE_ASSERT(PyType_Check(type));
  Py_INCREF(type);
  recarray = (PyTypeObject*)type;
}

void throw_array_conversion_error(PyObject* object, int flags, int rank_range, PyArray_Descr* descr) {
  ASSERT_IMPORTED();
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
  ASSERT_IMPORTED();

#if (NPY_FEATURE_VERSION == 0x00000009) && (NPY_ABI_VERSION == 0x01000009) && defined(_WIN64)
  // In NumPy 1.9.1 it appears that there are very few cases where it is possible to get arrays with the NPY_ARRAY_ALIGNED flag set
  // As a workaround we don't require NPY_ARRAY_ALIGNED as long as data is sufficiently aligned for our needs (SSE which needs 16 bytes)
  // On 64 bit windows it appears that arrays allocated by NumPy are already aligned to 16 bytes so this will always work
  // Note: Since conversion to uintptr_t could involve adding 3 (or some other invertible transform) this isn't technically portable behavior,
  //   however std::align is messy to use as a predicate since it mutates arguments. But I'm willing to bet this will work anywhere we care
  if (((uintptr_t)(static_cast<void*>(PyArray_DATA((PyArrayObject*)object))) % 16) == 0) {
    flags &= ~NPY_ARRAY_ALIGNED; // Clear the array aligned flag since we are aligned enough
  }
  else {
    // Alignment flag should trigger an exception from throw_array_conversion_error, but we emit a warning here to help track down the cause
    GEODE_WARNING("NumPy array alignment bug workaround failed!");
  }
#else
  // Maintainers of NumPy are aware of alignment issues and I think have a fix for the next release
  // The above workaround shouldn't be necessary if using an older or newer version of NumPy
  // However, instead of mysterious runtime errors we spit out a compile time error here
  // If you test this on a particular configuration you should add ifdefs to whitelist it here or use the workaround above
#  error Alignment for this version of NumPy has not been tested.
#endif

  const int rank = PyArray_NDIM((PyArrayObject*)object);
  const int min_rank = rank_range<0?-rank_range-1:rank_range,max_rank=rank_range<0?100:rank_range;
  if (!PyArray_CHKFLAGS((PyArrayObject*)object,flags) || min_rank>rank || rank>max_rank || !PyArray_EquivTypes(PyArray_DESCR((PyArrayObject*)object),descr))
    throw_array_conversion_error(object,flags,rank_range,descr);
}

#else // !defined(GEODE_PYTHON)

// CHAR_BIT isn't defined for some build configurations so we use __CHAR_BIT__ instead which seems to work in both clang and gcc
// It would probably be safe to just use 8 if this fails
#ifndef _WIN32
#define GEODE_CHAR_BIT __CHAR_BIT__
#else
#define GEODE_CHAR_BIT 8
#endif

// Modified from numpy/npy_common.h
typedef unsigned char npy_bool;
#define NPY_BITSOF_BOOL (sizeof(npy_bool)*GEODE_CHAR_BIT)
#define NPY_BITSOF_CHAR (sizeof(char)*GEODE_CHAR_BIT)
#define NPY_BITSOF_SHORT (sizeof(short)*GEODE_CHAR_BIT)
#define NPY_BITSOF_INT (sizeof(int)*GEODE_CHAR_BIT)
#define NPY_BITSOF_LONG (sizeof(long)*GEODE_CHAR_BIT)
#define NPY_BITSOF_LONGLONG (sizeof(long long)*GEODE_CHAR_BIT)
#define NPY_BITSOF_FLOAT (sizeof(float)*GEODE_CHAR_BIT)
#define NPY_BITSOF_DOUBLE (sizeof(double)*GEODE_CHAR_BIT)
#define NPY_BITSOF_LONGDOUBLE (sizeof(long double)*GEODE_CHAR_BIT)

// Lifted from numpy/ndarraytypes.h
enum NPY_TYPECHAR { NPY_GENBOOLLTR ='b',
                    NPY_SIGNEDLTR = 'i',
                    NPY_UNSIGNEDLTR = 'u',
                    NPY_FLOATINGLTR = 'f',
                    NPY_COMPLEXLTR = 'c'
};

#endif

Tuple<Array<uint8_t>,size_t> fill_numpy_header(int rank,const npy_intp* dimensions,int type_num) {
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
  const char endian = GEODE_ENDIAN==GEODE_LITTLE_ENDIAN ? '<'
                    : GEODE_ENDIAN==GEODE_BIG_ENDIAN    ? '>'
                                                        : 0;
  static_assert(endian,"Unknown endianness");

  // Construct header
  const char magic_version[8] = {(char)0x93,'N','U','M','P','Y',1,0};
  Array<uint8_t> header(256,uninit);
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
  GEODE_ASSERT(((len+1)&15)==0);
  header[len++] = '\n';
  GEODE_ASSERT((len&15)==0);
  uint16_t header_len = uint16_t(len-10);
  GEODE_ASSERT(header_len==len-10);
  if (GEODE_ENDIAN == GEODE_BIG_ENDIAN) {
    // Switch header_len to little endian
    swap(((char*)&header_len)[0],((char*)&header_len)[1]);
  }
  memcpy(base+8,&header_len,2);
  header.resize(len);
  return tuple(header,bytes*total_size);
}

void write_numpy(const string& filename,int rank,const npy_intp* dimensions,int type_num,void* data) {
  // Make header
  const auto H = fill_numpy_header(rank,dimensions,type_num);

  // Write npy file
  FILE* file = fopen(filename.c_str(),"wb");
  if(!file) throw OSError("Can't open "+filename+" for writing");
  fwrite(H.x.data(),1,H.x.size(),file);
  fwrite(data,1,H.y,file);
  fclose(file);
}

}
using namespace geode;

void wrap_numpy() {
#ifdef GEODE_PYTHON
  GEODE_FUNCTION(_set_recarray_type)
#endif
}
