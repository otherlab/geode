//#####################################################################
// Function wrap_property
//#####################################################################
//
// Wrap a C++ accessor function as a python property.  It is normally used indirectly through Class<T>.
//
//#####################################################################
#pragma once

#include <other/core/python/config.h>
#include <other/core/python/exceptions.h>
#include <other/core/python/from_python.h>
#include <other/core/python/to_python.h>
#include <other/core/utility/config.h>
#include <boost/mpl/bool.hpp>
namespace other {

namespace mpl = boost::mpl;
using std::exception;

OTHER_EXPORT PyObject* wrap_property_helper(PyTypeObject* type,const char* name,getter get_wrapper,setter set_wrapper,void* get);

template<class T,class A> static PyObject*
property_get_wrapper(PyObject* self, void* get) {
  try {
    typedef A (T::*Get)() const;
    return to_python((((T*)(self+1))->*(*(Get*)get))());
  } catch (const exception& error) {
    set_python_exception(error);
    return 0;
  }
}

template<class T,class B,class R> static int
property_set_wrapper(PyObject* self, PyObject* value, void* getset) {
  try {
    typedef int (T::*Get)() const; // It's safe to pretend the return type is int, since it doesn't affect the size
    typedef R (T::*Set)(B);
    (((T*)(self+1))->*(*(Set*)((char*)getset+sizeof(Get))))(from_python<B>(value));
    return 0;
  } catch (const exception& error) {
    set_python_exception(error);
    return -1;
  }
}

template<class T,class A> static PyObject*
wrap_property(const char* name,A (T::*get)() const) {
  typedef A (T::*Get)() const;
  return wrap_property_helper(&T::pytype,name,property_get_wrapper<T,A>,0,(void*)new Get(get));
}

template<class T,class A,class B,class R> static PyObject*
wrap_property(const char* name,A (T::*get)() const,R (T::*set)(B)) {
  typedef A (T::*Get)() const;
  typedef R (T::*Set)(B);
  char* getset = (char*)malloc(sizeof(Get)+sizeof(Set));
  *(Get*)getset = get;
  *(Set*)(getset+sizeof(get)) = set;
  return wrap_property_helper(&T::pytype,name,property_get_wrapper<T,A>,property_set_wrapper<T,B,R>,getset);
}

}
