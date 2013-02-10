//#####################################################################
// Module Vectors
//#####################################################################
#include <other/core/vector/Matrix.h>
#include <other/core/vector/Vector.h>
#include <other/core/array/NdArray.h>
#include <other/core/python/wrap.h>
#include <sstream>
using namespace other;
namespace {

using std::istringstream;

#ifdef OTHER_PYTHON

static Vector<int,3> vector_test(const Vector<int,3>& vector) {
  return vector;
}

static Array<Matrix<real,4>> matrix_test(Array<Matrix<real,4> > array) {
  Matrix<real,4> A = Matrix<real,4>::identity_matrix();
  OTHER_ASSERT(array[1]==A);
  A(1,2) = 3;
  OTHER_ASSERT(array[0]==A);
  Array<Matrix<real,4> > test_matrix;
  test_matrix = array;
  return test_matrix;
}

double min_magnitude_python(NdArray<const real> array) {
  switch (array.rank()) {
    case 1: return array.flat.min_magnitude();
    case 2:
      switch (array.shape[1]) {
        case 0: return 0;
        case 1: return array.flat.min_magnitude();
        case 2: return RawArray<const Vector<real,2> >(array.shape[0],(const Vector<real,2>*)array.data()).min_magnitude();
        case 3: return RawArray<const Vector<real,3> >(array.shape[0],(const Vector<real,3>*)array.data()).min_magnitude();
        default: PyErr_Format(PyExc_ValueError,"min_magnitude not implemented for shape (%d,%d)",array.shape[0],array.shape[1]);
      }
    default: PyErr_Format(PyExc_ValueError,"min_magnitude: expected rank 1 or 2, got %d",array.rank());
  }
  throw_python_error();
}

double max_magnitude_python(NdArray<const real> array) {
  switch (array.rank()) {
    case 1: return array.flat.max_magnitude();
    case 2:
      switch (array.shape[1]) {
        case 0: return 0;
        case 1: return array.flat.max_magnitude();
        case 2: return RawArray<const Vector<real,2> >(array.shape[0],(const Vector<real,2>*)array.data()).max_magnitude();
        case 3: return RawArray<const Vector<real,3> >(array.shape[0],(const Vector<real,3>*)array.data()).max_magnitude();
        default: PyErr_Format(PyExc_ValueError,"max_magnitude not implemented for shape (%d,%d)",array.shape[0],array.shape[1]);
      }
    default: PyErr_Format(PyExc_ValueError,"max_magnitude: expected rank 1 or 2, got %d",array.rank());
  }
  throw_python_error();
}

void vector_stream_test() {
  istringstream ss("[1,2,3]a"); 
  Vector<int,3> v;
  ss >> v >> expect('a');
  ss.str("[1,2,3)");
  try {
    ss >> v;
    OTHER_ASSERT(false);
  } catch (const ValueError&) {}
}

#endif
}

void wrap_vector() {
  OTHER_WRAP(matrix)
  OTHER_WRAP(rotation)
  OTHER_WRAP(frame)
  OTHER_WRAP(sparse_matrix)
  OTHER_WRAP(solid_matrix)
  OTHER_WRAP(register)
  OTHER_WRAP(test)

#ifdef OTHER_PYTHON
  OTHER_FUNCTION_2(min_magnitude,min_magnitude_python)
  OTHER_FUNCTION_2(max_magnitude,max_magnitude_python)

  // for testing purposes
  OTHER_FUNCTION(vector_test)
  OTHER_FUNCTION(matrix_test)
  OTHER_FUNCTION(vector_stream_test)
#endif
}
