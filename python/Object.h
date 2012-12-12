//#####################################################################
// Class Object
//#####################################################################
//
// Object is a base class for C++ objects which are also python objects.
//
// Since python objects are reference counted, instances of Object can be owned only through Ref<T> or Ptr<T>,
// which manage the reference count.  To allocate a new object of type T, use new_<T>(...), which returns a Ref.
//
// Classes derived from object are noncopyable by default.
//
// Note that the python header is stored immediately _before_ the start of the C++ object, since C++ insists on
// putting the vtable at the start of objects.  Thus a PyObject* p can be converted to a C++ T* via (T*)(p+1),
// and a T* q can be converted to PyObject* via (PyObject*)q-1.
//
// Since they are exposed to python directly by casting T* to PyObject*, it is impossible to preserve const
// semantics across the C++ python layer: any information about immutability must apply to the object itself,
// not each reference to the object.
//
// Descendants of Object must use OTHER_DECLARE_TYPE / OTHER_DEFINE_TYPE to expose information about the derived
// type to the python wrapping layer.
//
//#####################################################################
#pragma once

#include <other/core/python/forward.h>
#include <other/core/python/new.h>
#include <other/core/utility/config.h>
#include <boost/mpl/void.hpp>
#include <boost/utility/enable_if.hpp>

namespace other {

namespace mpl = boost::mpl;

class OTHER_CORE_CLASS_EXPORT Object {
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
  typedef Object Base; // the hierarchy stops here
private:
  Object(const Object&); // noncopyable by default
  void operator=(const Object&); // noncopyable by default
protected:
  Object();
public:
  virtual ~Object();
};

// Helper for derived classes which are simple wrappers around other real classes
template<class T> struct GetSelf {
  typedef T type;
  static T* get(PyObject* object) {
    BOOST_MPL_ASSERT((boost::is_base_of<Object,T>));
    return (T*)(object+1);
  }
};

// Conversion from T& for python types
template<class T> static inline
typename boost::enable_if<boost::is_base_of<Object,T>,PyObject*>::type
to_python(T& value) {
  PyObject* object = (PyObject*)&value-1;
  OTHER_INCREF(object);
  return object;
}

}
