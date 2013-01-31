#include <other/core/geometry/Segment2d.h>
#include <other/core/geometry/Segment3d.h>
#include <other/core/vector/convert.h>
#include <other/core/structure/Tuple.h>

namespace other{
#ifdef OTHER_PYTHON

template<class TV> PyObject* to_python(const Segment<TV>& seg) {
  return to_python(tuple(seg.x0,seg.x1));
}

#else
#define SEGMENT_CONVERSIONS(...)
#endif

}
