// Class Box

#include <other/core/geometry/Box.h>
namespace other {

#ifndef WIN32
template Box<real> bounding_box(const RawArray<const real>&);
template Box<Vector<real,2>> bounding_box(const RawArray<const Vector<real,2>>&);
template Box<Vector<real,3>> bounding_box(const RawArray<const Vector<real,3>>&);
#endif

}
