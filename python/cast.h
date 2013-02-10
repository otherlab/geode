//#####################################################################
// Type checking and cast utilities for Python objects
//#####################################################################
#pragma once

#include <other/core/python/Object.h>
namespace other {

#ifdef OTHER_PYTHON

template<class T> static inline bool is_instance(PyObject* object) {
  return PyObject_IsInstance(object,(PyObject*)&T::pytype)!=0;
}

#else // Non-python stub.  For now, this detects only exact matches.

template<class T> static inline bool is_instance(PyObject* object) {
  return object->ob_type==&T::pytype;
}

#endif

template<class T> struct PythonCast;
template<class T> struct PythonCast<T*> { static T* cast(PyObject* object) {
  return is_instance<T>(object) ? GetSelf<T>::get(object) : 0;
}};
template<class T> struct PythonCast<T&> { static T& cast(PyObject* object) {
#ifdef OTHER_PYTHON
  if (!boost::is_same<T,Object>::value && &T::pytype==&T::Base::pytype)
    unregistered_python_type(object,&T::pytype,typeid(T));
#endif
  if (is_instance<T>(object))
    return *GetSelf<T>::get(object);
  throw_type_error(object,&T::pytype);
}};


template<class T> static inline T python_cast(PyObject* object) {
  return PythonCast<T>::cast(object);
}

template<class T> static inline T python_cast(const Object& object) {
  return PythonCast<T>::cast(to_python(object));
}

}
