// Header wrap: various functions for adding objects to the current module
#pragma once

#include <other/core/python/config.h>
#include <other/core/utility/config.h>
#ifdef OTHER_PYTHON
#include <other/core/python/wrap_function.h>
#endif
namespace other {
namespace python {

// Steal reference to object and add it to the current module.
// Defined in module.cpp.
OTHER_CORE_EXPORT void add_object(const char* name, other::PyObject* object);

template<class T> static inline void add_object(const char* name, const T& object) {
#ifdef OTHER_PYTHON
  add_object(name,to_python(object));
#endif
}

template<class Function> static inline void function(const char* name, Function function) {
#ifdef OTHER_PYTHON
  add_object(name,wrap_function(name,function));
#endif
}

#define OTHER_OBJECT(name) ::other::python::add_object(#name,name);
#define OTHER_OBJECT_2(name,object) ::other::python::add_object(#name,object);

#define OTHER_FUNCTION(name) ::other::python::function(#name,name);
#define OTHER_FUNCTION_2(name,...) ::other::python::function(#name,__VA_ARGS__);

#define OTHER_OVERLOADED_FUNCTION_2(type,name,function_) ::other::python::function(name,(type)function_);

#define OTHER_OVERLOADED_FUNCTION(type,function_) OTHER_OVERLOADED_FUNCTION_2(type,#function_,function_);

#ifndef OTHER_WRAP
#ifdef OTHER_PYTHON
#define OTHER_WRAP(name) extern void wrap_##name();wrap_##name();
#else
#define OTHER_WRAP(name)
#endif
#endif

}
}
