// Numpy definitions without Numpy dependencies

#include <geode/utility/numpy.h>
#include <boost/detail/endian.hpp>
#include <boost/cstdint.hpp>
#include <stdio.h>
namespace geode {

// Lifted from numpy/ndarraytypes.h
enum NPY_TYPECHAR { NPY_GENBOOLLTR ='b',
                    NPY_SIGNEDLTR = 'i',
                    NPY_UNSIGNEDLTR = 'u',
                    NPY_FLOATINGLTR = 'f',
                    NPY_COMPLEXLTR = 'c'
};

// We're going to use this for I/O, so if char isn't 8 bits we're hosed anyways.
#define GEODE_CHAR_BIT 8

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

size_t fill_numpy_header_helper(Array<uint8_t>& header, RawArray<const long> shape, const int type) {
  // Get dtype info
  int bits;
  char letter;
  switch (type) {
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
  header.clear();
  header.resize(256,uninit);
  char* const base = (char*)header.data();
  memcpy(base,magic_version,8);
  int len = 10;
  len += sprintf(base+len,"{'descr': '%c%c%d', 'fortran_order': False, 'shape': (",endian,letter,bytes);
  size_t total_size = 1;
  const int rank = shape.size();
  for (int i=0;i<rank;i++) {
    total_size *= shape[i];
    len += sprintf(base+len,"%ld%s",shape[i],rank==1||i<rank-1?",":"");
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
#ifdef BOOST_BIG_ENDIAN
  // Switch header_len to little endian
  swap(((char*)&header_len)[0],((char*)&header_len)[1]);
#endif
  memcpy(base+8,&header_len,2);
  header.resize(len);
  return bytes*total_size;
}

void write_numpy_helper(const string& filename, RawArray<const long> shape, const int type, const void* data) {
  // Fill header
  Array<uint8_t> header;
  const size_t data_size = fill_numpy_header_helper(header,shape,type);

  // Write npy file
  FILE* file = fopen(filename.c_str(),"wb");
  if(!file) throw OSError("Can't open "+filename+" for writing");
  fwrite(header.data(),1,header.size(),file);
  fwrite(data,1,data_size,file);
  fclose(file);
}

Array<uint8_t> array_write_test(const string& filename, RawArray<const real,2> array) {
  write_numpy(filename,array);
  Array<uint8_t> header;
  fill_numpy_header(header,array);
  return header;
}

}
