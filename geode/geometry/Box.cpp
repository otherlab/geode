// Class Box

#include <geode/geometry/Box.h>
namespace geode {

#ifndef _WIN32
template Box<real> bounding_box(const RawArray<const real>&);
template Box<Vector<real,2>> bounding_box(const RawArray<const Vector<real,2>>&);
template Box<Vector<real,3>> bounding_box(const RawArray<const Vector<real,3>>&);
#endif

}
