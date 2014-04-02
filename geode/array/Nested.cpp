//#####################################################################
// Class Nested
//#####################################################################
#include <geode/array/Nested.h>
namespace geode {

Array<int> nested_array_offsets(RawArray<const int> lengths) {
  Array<int> offsets(lengths.size()+1,uninit);
  offsets[0] = 0;
  for (int i=0;i<lengths.size();i++) {
    GEODE_ASSERT(lengths[i]>=0);
    offsets[i+1] = offsets[i]+lengths[i];
  }
  return offsets;
}

}
