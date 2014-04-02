//#####################################################################
// Class Array
//#####################################################################
#include <geode/array/Array.h>
#include <geode/array/Nested.h>
#include <geode/utility/exceptions.h>
#include <geode/vector/Vector.h>
namespace geode {

void out_of_bounds(const type_info& type, const int size, const int i) {
  throw IndexError(format("Array index out of bounds: type %s, len %d, index %d",type.name(),size,i));
}

// For testing purposes:

Array<int> empty_array_test() {
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
  for (int i=0;i<5;i++)
    a0.append(i);

  // Some very basic tests to check that append/extend aren't drastically broken
  Nested<int,false> n0;
  GEODE_ASSERT(n0.size() == 0);
  n0.append(a0);
  GEODE_ASSERT(n0.back().size() == a0.size());
  GEODE_ASSERT(n0.size() == 1);
  for (int i=0;i<a0.size();i++)
    GEODE_ASSERT(n0.flat[i] == a0[i]);
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

}
