//#####################################################################
// Numpy interface functions
//#####################################################################
#include <other/core/python/numpy.h>
#include <numpy/npy_common.h>
#include <boost/detail/endian.hpp>
#include <boost/cstdint.hpp>
namespace other {

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

void throw_array_conversion_error(PyObject* object, int flags, int rank_range, PyArray_Descr* descr) {
  if (!PyArray_EquivTypes(PyArray_DESCR((PyArrayObject*)object), descr))
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

size_t fill_numpy_header(Array<uint8_t>& header,int rank,const npy_intp* dimensions,int type_num) {
  // Get dtype info
  int bits;
  char letter;
  switch (type_num) {
    #define CASE(T,K) case NPY_##T:bits=NPY_BITSOF_##T;letter=NPY_##K##LTR;break;
    #define NPY_BITSOF_BYTE 8
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
  int bytes = bits/8;

  // Endianness
#if defined(BOOST_LITTLE_ENDIAN)
  const char endian = '<';
#elif defined(BOOST_BIG_ENDIAN)
  const char endian = '>';
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
  memcpy(base+8,&header_len,2);
  header.resize(len);
  return bytes*total_size;
}

void write_numpy(const string& filename,int rank,const npy_intp* dimensions,int type_num,void* data) {
  // Fill header
  Array<uint8_t> header;
  size_t data_size = fill_numpy_header(header,rank,dimensions,type_num);

  // Write npy file
  FILE* file = fopen(filename.c_str(),"w");
  if(!file) throw OSError("Can't open "+filename+" for writing");
  fwrite(header.data(),1,header.size(),file);
  fwrite(data,1,data_size,file);
  fclose(file);
}

}
