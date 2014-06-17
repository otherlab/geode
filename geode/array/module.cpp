//#####################################################################
// Module Arrays
//#####################################################################
#include <geode/array/Array2d.h>
#include <geode/array/Nested.h>
#include <geode/python/numpy.h>
#include <geode/python/wrap.h>
using namespace geode;

namespace {

Array<int> empty_array() {
  return Array<int>();
}

Array<int> array_test(Array<int> array, int resize) {
  Array<int> test;
  test = array;
  if(resize>=0)
    test.resize(resize);
  return test;
}

Array<const int> const_array_test(Array<const int> array) {
  Array<const int> test;
  test = array;
  return test;
}

void nested_test() {
  Array<int> a0;
  for(int i = 0; i < 5; ++i) {
    a0.append(i);
  }

  // Some very basic tests to check that append/extend aren't drastically broken
  Nested<int, false> n0;
  GEODE_ASSERT(n0.size() == 0);
  n0.append(a0);
  GEODE_ASSERT(n0.back().size() == a0.size());
  GEODE_ASSERT(n0.size() == 1);
  for(int i = 0; i < a0.size(); ++i) {
    GEODE_ASSERT(n0.flat[i] == a0[i]);
  }
  n0.append_to_back(12);
  GEODE_ASSERT(n0.size() == 1);
  GEODE_ASSERT(n0.back().size() == a0.size() + 1);
  GEODE_ASSERT(n0.flat.back() == 12);

  n0.extend_back(a0);
  GEODE_ASSERT(n0.size() == 1);
  GEODE_ASSERT(n0.back().size() == 2*a0.size() + 1);

  n0.append(a0);
  GEODE_ASSERT(n0.size() == 2);
  GEODE_ASSERT(n0.back().size() == a0.size());

  // Check that concatenate works
  Nested<int> n1 = concatenate(n0.freeze());
  GEODE_ASSERT(n0.freeze() == n1);

  Nested<int> n2 = concatenate(n0.freeze(),n1);
  GEODE_ASSERT(n2.total_size() == n0.total_size() + n1.total_size());
  GEODE_ASSERT(n2.size() == n0.size() + n1.size());
}

Nested<const int> nested_convert_test(Nested<const int> a) {
  return a;
}

#ifdef GEODE_PYTHON

ssize_t base_refcnt(PyObject* array) {
  GEODE_ASSERT(PyArray_Check(array));
  PyObject* base = PyArray_BASE((PyArrayObject*)array);
  return base?base->ob_refcnt:0;
}

Tuple<Array<uint8_t>,size_t> array_write_test(const string& filename, RawArray<const real,2> array) {
  write_numpy(filename,array);
  return fill_numpy_header(array);
}

#endif

}

void wrap_array() {
  GEODE_WRAP(nested_array)
  GEODE_WRAP(stencil)

  // for testing purposes
  GEODE_FUNCTION(empty_array)
  GEODE_FUNCTION(array_test)
  GEODE_FUNCTION(nested_test)
  GEODE_FUNCTION(nested_convert_test)
  GEODE_FUNCTION(const_array_test)
#ifdef GEODE_PYTHON
  GEODE_FUNCTION(base_refcnt)
  GEODE_FUNCTION(array_write_test)
#endif
}
