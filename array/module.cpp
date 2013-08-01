//#####################################################################
// Module Arrays
//#####################################################################
#include <other/core/array/Array2d.h>
#include <other/core/array/Nested.h>
#include <other/core/python/numpy.h>
#include <other/core/python/wrap.h>
using namespace other;

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
  OTHER_ASSERT(n0.size() == 0);
  n0.append(a0);
  OTHER_ASSERT(n0.back().size() == a0.size());
  OTHER_ASSERT(n0.size() == 1);
  for(int i = 0; i < a0.size(); ++i) {
    OTHER_ASSERT(n0.flat[i] == a0[i]);
  }
  n0.append_to_back(12);
  OTHER_ASSERT(n0.size() == 1);
  OTHER_ASSERT(n0.back().size() == a0.size() + 1);
  OTHER_ASSERT(n0.flat.back() == 12);

  n0.extend_back(a0);
  OTHER_ASSERT(n0.size() == 1);
  OTHER_ASSERT(n0.back().size() == 2*a0.size() + 1);

  n0.append(a0);
  OTHER_ASSERT(n0.size() == 2);
  OTHER_ASSERT(n0.back().size() == a0.size());

  // Check that concatenate works
  Nested<int> n1 = concatenate(n0.freeze());
  OTHER_ASSERT(n0.freeze() == n1);

  Nested<int> n2 = concatenate(n0.freeze(),n1);
  OTHER_ASSERT(n2.total_size() == n0.total_size() + n1.total_size());
  OTHER_ASSERT(n2.size() == n0.size() + n1.size());
}

#ifdef OTHER_PYTHON

ssize_t base_refcnt(PyObject* array) {
  OTHER_ASSERT(PyArray_Check(array));
  PyObject* base = PyArray_BASE((PyArrayObject*)array);
  return base?base->ob_refcnt:0;
}

Array<uint8_t> array_write_test(const string& filename, RawArray<const real,2> array) {
  write_numpy(filename,array);
  Array<uint8_t> header;
  fill_numpy_header(header,array);
  return header;
}

#endif

}

void wrap_array() {
  OTHER_WRAP(nested_array)

  // for testing purposes
  OTHER_FUNCTION(empty_array)
  OTHER_FUNCTION(array_test)
  OTHER_FUNCTION(nested_test)
  OTHER_FUNCTION(const_array_test)
#ifdef OTHER_PYTHON
  OTHER_FUNCTION(base_refcnt)
  OTHER_FUNCTION(array_write_test)
#endif
}
