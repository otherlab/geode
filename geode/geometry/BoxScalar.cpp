//#####################################################################
// Class Box<T>
//#####################################################################
#include <geode/geometry/BoxScalar.h>
#include <geode/python/exceptions.h>
#include <geode/structure/Tuple.h>
#include <geode/vector/Vector.h>
namespace geode {

#ifdef GEODE_PYTHON

template<class T> PyObject* to_python(const Box<T>& self) {
  return to_python(tuple(self.min,self.max));
}

template<class T> Box<T> FromPython<Box<T> >::convert(PyObject* object) {
  const auto extents = from_python<Tuple<T,T>>(object);
  return Box<T>(extents.x,extents.y);
}

#define INSTANTIATE(T) \
  template GEODE_CORE_EXPORT PyObject* to_python(const Box<T>&); \
  template GEODE_CORE_EXPORT Box<T> FromPython<Box<T> >::convert(PyObject*);
INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(int64_t)

#endif

}
