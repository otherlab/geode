// Header wrap: various functions for adding objects to the current module
#pragma once

#include <geode/python/config.h>
#include <geode/utility/config.h>
#ifdef GEODE_PYTHON
#include <geode/python/wrap_function.h>
#endif
namespace geode {
namespace python {

// Steal reference to object and add it to the current module.
// Defined in module.cpp.
GEODE_CORE_EXPORT void add_object(const char* name, geode::PyObject* object);

template<class T> static inline void add_object(const char* name, const T& object) {
#ifdef GEODE_PYTHON
  add_object(name,to_python(object));
#endif
}

template<class Function> static inline void function(const char* name, Function function) {
#ifdef GEODE_PYTHON
  add_object(name,wrap_function(name,function));
#endif
}

#define GEODE_OBJECT(name) ::geode::python::add_object(#name,name);
#define GEODE_OBJECT_2(name,object) ::geode::python::add_object(#name,object);

#define GEODE_FUNCTION(name) ::geode::python::function(#name,name);
#define GEODE_FUNCTION_2(name,...) ::geode::python::function(#name,__VA_ARGS__);

#define GEODE_OVERLOADED_FUNCTION_2(type,name,function_) ::geode::python::function(name,(type)function_);

#define GEODE_OVERLOADED_FUNCTION(type,function_) GEODE_OVERLOADED_FUNCTION_2(type,#function_,function_);

#ifndef GEODE_WRAP
#ifdef GEODE_PYTHON
#define GEODE_WRAP(name) extern void wrap_##name();wrap_##name();
#else
#define GEODE_WRAP(name)
#endif
#endif

}
}
