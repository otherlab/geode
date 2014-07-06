// Convert to and from python if possible, otherwise throw an exception
#pragma once

#include <geode/python/to_python.h>
#include <geode/python/from_python.h>
#include <geode/utility/forward.h>
#include <geode/utility/type_traits.h>
namespace geode {

#ifdef GEODE_PYTHON

GEODE_CORE_EXPORT void set_to_python_failed(const type_info& type);
GEODE_CORE_EXPORT void GEODE_NORETURN(from_python_failed(PyObject* object, const type_info& type));

template<class T> static inline typename enable_if<has_to_python<T>,PyObject*>::type try_to_python(const T& x) {
  return to_python(x);
}

template<class T> static inline typename disable_if<has_to_python<T>,PyObject*>::type try_to_python(const T& x) {
  set_to_python_failed(typeid(T));
  return 0;
}

template<class T> static inline typename enable_if<has_from_python<T>,T>::type try_from_python(PyObject* object) {
  return from_python<T>(object);
}

template<class T> static inline typename disable_if<has_from_python<T>,T>::type try_from_python(PyObject* object) {
  from_python_failed(object,typeid(T));
}

#endif

}
