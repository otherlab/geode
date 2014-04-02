//#####################################################################
// Function wrap_field
//#####################################################################
//
// Constructs a python descriptor for a C++ field.  It is normally used indirectly through Class<T>.
//
// By default, the property is readonly if the C++ field is const, and writeable otherwise.  Fields exposed
// this way should be efficiently shareable or copyable, since __set__ is implemented by converting the python
// object to the appropriate type and using operator=.  __get__ and __set__ use to_python and from_python,
// respectively.  See to_python.h and from_python.h for details.
//
//#####################################################################
#pragma once

#include <geode/python/config.h>
#include <geode/utility/exceptions.h>
#include <geode/utility/type_traits.h>
namespace geode {

using std::exception;

GEODE_CORE_EXPORT PyObject* wrap_field_helper(PyTypeObject* type,const char* name,size_t offset,getter get,setter set);

// use an unnamed namespace since instances of wrap_field never need to be shared
namespace {

template<class T,class S> PyObject*
get_wrapper(PyObject* self, void* offset) {
  try {
    return to_python(*(const S*)((char*)GetSelf<T>::get(self)+(size_t)offset));
  } catch (const exception& error) {
    set_python_exception(error);
    return 0;
  }
}

template<class T,class S> int
set_wrapper(PyObject* self, PyObject* value, void* offset) {
  try {
    *(S*)((char*)GetSelf<T>::get(self)+(size_t)offset) = from_python<S>(value);
    return 0;
  } catch (const exception& error) {
    set_python_exception(error);return -1;
  }
}

template<class T,class B,class S> PyObject*
wrap_field(const char* name, const S B::* field) { // const fields
  // On windows, .* sends null pointers to null pointers, so we use 64 instead.
  size_t offset = (char*)&((*(typename GetSelf<T>::type*)64).*field)-(char*)64;
  return wrap_field_helper(&T::pytype,name,offset,get_wrapper<T,S>,0);
}

template<class T,class B,class S> typename disable_if<is_const<S>,PyObject*>::type
wrap_field(const char* name, S B::* field) { // nonconst fields
  // On windows, .* sends null pointers to null pointers, so we use 64 instead.
  size_t offset = (char*)&((*(typename GetSelf<T>::type*)64).*field)-(char*)64;
  return wrap_field_helper(&T::pytype,name,offset,get_wrapper<T,S>,set_wrapper<T,S>);
}

template<class T,class S> PyObject*
wrap_field(const char* name,const S* field) { // static const fields
  return to_python(*field);
}

}
}
