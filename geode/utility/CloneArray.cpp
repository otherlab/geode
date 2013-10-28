#include <geode/utility/CloneArray.h>
namespace geode {

CloneArray<CloneableBase>::
CloneArray(const CloneableBase& template_object, const int count)
  : sizeof_clone(template_object.sizeof_clone()), count(count) {
  assert(count>=0);
  data = (char*)malloc(sizeof_clone*count);
  int i = 0;
  try {
    for(;i<count;i++)
      template_object.placement_clone(&(*this)(i));
  } catch (...) {
    for (int j=0;j<i;j++)
      (*this)(j).~CloneableBase();
    free(data);
    throw;
  }
}

CloneArray<CloneableBase>::
~CloneArray() {
  for(int i=0;i<count;i++)
      (*this)(i).~CloneableBase();
  free(data);
}

}
