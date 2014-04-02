// Functions for Python use

#include <geode/array/NdArray.h>
#include <geode/array/Nested.h>
#include <geode/vector/Matrix.h>
#include <geode/vector/Vector.h>
namespace geode {

typedef real T;
using std::istringstream;

Vector<int,3> vector_test(const Vector<int,3>& vector) {
  return vector;
}

Array<Matrix<T,4>> matrix_test(Array<Matrix<T,4>> array) {
  Matrix<T,4> A = Matrix<T,4>::identity_matrix();
  GEODE_ASSERT(array[1]==A);
  A(1,2) = 3;
  GEODE_ASSERT(array[0]==A);
  Array<Matrix<T,4>> test_matrix;
  test_matrix = array;
  return test_matrix;
}

double min_magnitude_python(NdArray<const T> array) {
  switch (array.rank()) {
    case 1: return array.flat.min_magnitude();
    case 2:
      switch (array.shape[1]) {
        case 0: return 0;
        case 1: return array.flat.min_magnitude();
        case 2: return RawArray<const Vector<T,2>>(array.shape[0],(const Vector<T,2>*)array.data()).min_magnitude();
        case 3: return RawArray<const Vector<T,3>>(array.shape[0],(const Vector<T,3>*)array.data()).min_magnitude();
        default: throw ValueError(format("min_magnitude not implemented for shape (%d,%d)",
                                         array.shape[0],array.shape[1]));
      }
    default: throw ValueError(format("min_magnitude: expected rank 1 or 2, got %d",array.rank()));
  }
}

double max_magnitude_python(NdArray<const T> array) {
  switch (array.rank()) {
    case 1: return array.flat.max_magnitude();
    case 2:
      switch (array.shape[1]) {
        case 0: return 0;
        case 1: return array.flat.max_magnitude();
        case 2: return RawArray<const Vector<T,2>>(array.shape[0],(const Vector<T,2>*)array.data()).max_magnitude();
        case 3: return RawArray<const Vector<T,3>>(array.shape[0],(const Vector<T,3>*)array.data()).max_magnitude();
        default: throw ValueError(format("max_magnitude not implemented for shape (%d,%d)",
                                         array.shape[0],array.shape[1]));
      }
    default: throw ValueError(format("max_magnitude: expected rank 1 or 2, got %d",array.rank()));
  }
}

void vector_stream_test() {
  istringstream ss("[1,2,3]a");
  Vector<int,3> v;
  ss >> v >> expect('a');
  ss.str("[1,2,3)");
  try {
    ss >> v;
    GEODE_ASSERT(false);
  } catch (const ValueError&) {}
}

}
