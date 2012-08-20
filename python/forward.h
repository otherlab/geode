//#####################################################################
// Header Python/Forward
//#####################################################################
#pragma once

struct _object;
typedef _object PyObject;
struct _typeobject;
typedef _typeobject PyTypeObject;

#include <other/core/python/config.h>
#include <other/core/utility/config.h>

namespace other {

class Object;
struct Buffer;
template<class T=PyObject> class Ref;
template<class T=PyObject> class Ptr;

template<class T,class Enable=void> struct FromPython; // from_python<T> isn't defined for types by default

template<class T,class... Args> static inline Ref<T> new_(Args&&... args);
template<class T,class... Args> static PyObject* wrapped_constructor(PyTypeObject* type,PyObject* args,PyObject* kwds);

// Macro to declare new_ as a friend
#define OTHER_NEW_FRIEND \
  template<class _T,class... _Args> friend other::Ref<_T> other::new_(_Args&&... args); \
  template<class _T,class... _Args> friend PyObject* ::other::wrapped_constructor(PyTypeObject* type,PyObject* args,PyObject* kwds);

// Should appear at the beginning of all mixed python/C++ classes, after public:
#define OTHER_DECLARE_TYPE \
  OTHER_NEW_FRIEND \
  OTHER_EXPORT static PyTypeObject pytype;

} // namespace other
