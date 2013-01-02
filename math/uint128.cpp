//#####################################################################
// Class uint128_t
//#####################################################################
#include <other/core/math/uint128.h>
#include <other/core/python/stl.h>
#include <other/core/python/module.h>
#include <other/core/utility/format.h>
#include <boost/static_assert.hpp>
#include <iostream>
namespace other{

using std::cout;

string str(uint128_t n) {
  const auto lo = cast_uint128<uint64_t>(n),
             hi = cast_uint128<uint64_t>(n>>64);
  // For now, we lazily produce hexadecimal to avoid having to divide.
  return hi ? format("0x%llx%016llx",hi,lo) : format("0x%llx",lo);
}

ostream& operator<<(ostream& output, uint128_t n) {
  return output << str(n);
}

#ifdef OTHER_PYTHON

static PyObject* p64;

PyObject* to_python(uint128_t n) {
  PyObject *lo=0,*hi=0,*shi=0,*r=0;
  lo = PyLong_FromUnsignedLongLong(uint64_t(n&~0L));
  uint64_t nhi = (n>>64)&~0L;
  if (!nhi || !lo) return lo;
  hi = PyLong_FromUnsignedLongLong(nhi);
  if (!hi) goto done;
  shi = PyNumber_Lshift(hi,p64);
  if (!shi) goto done;
  r = PyNumber_Add(shi,lo);
done:
  Py_XDECREF(lo);
  Py_XDECREF(hi);
  Py_XDECREF(shi);
  return r;
}

uint128_t FromPython<uint128_t>::convert(PyObject* object) {
  BOOST_STATIC_ASSERT(sizeof(long long)==sizeof(uint64_t));
  PyObject* n = PyNumber_Int(object);
  if (!n) throw_python_error();
  uint64_t lo = PyInt_AsUnsignedLongLongMask(n);
  if (lo==(uint64_t)-1 && PyErr_Occurred()){Py_DECREF(n);throw_python_error();}
  PyObject* phi = PyNumber_Rshift(n,p64);
  Py_DECREF(n);
  if (!phi) throw_python_error();
  uint64_t hi;
  if (PyInt_Check(phi)) {
    long shi = PyInt_AS_LONG(phi);
    if (shi < 0) {
      hi = -1;
      PyErr_SetString(PyExc_OverflowError,"can't convert negative number to uint128_t");
    } else
      hi = shi;
  } else
    hi = PyLong_AsUnsignedLongLong(phi);
  Py_DECREF(phi);
  if (hi==(uint64_t)-1 && PyErr_Occurred()) throw_python_error();
  return (uint128_t(hi)<<64)|lo;
}

static std::vector<uint128_t> uint128_test(uint128_t x,uint128_t y) {
  // Test shifts
  static const int shifts[] = {0,1,63,64,65,127};
  for (const int s : shifts) {
    const auto p = uint128_t(1)<<s;
    OTHER_ASSERT((x<<s)==x*p);
    OTHER_ASSERT(((x>>s)<<s)==(x&~(p-1)));
  }
  // Test other operations
  std::vector<uint128_t> r;
  r.push_back((uint64_t)-7);
  r.push_back(-7);
  r.push_back(x);
  r.push_back(x+y);
  r.push_back(x-y);
  r.push_back(x*y);
  r.push_back(x<<5);
  r.push_back(x>>7);
  return r;
}

#endif

}
using namespace other;

void wrap_uint128() {
#ifdef OTHER_PYTHON
  p64 = PyInt_FromLong(64);
  if (!p64) throw_python_error();

  OTHER_FUNCTION(uint128_test)
  OTHER_FUNCTION_2(uint128_str_test,static_cast<string(*)(uint128_t)>(str))
#endif
}
