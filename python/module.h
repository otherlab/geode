//#####################################################################
// Header Module
//#####################################################################
//
// Macro to declare a python module initialization function.  This does two things:
//
// 1. Catches and translates any C++ exceptions into python.
// 2. Sets the current_module variable for use in Class and other wrapper helpers.
//
// If the module is named blah, the shared library should be blah.so, not libblah.so.
//
//#####################################################################
#pragma once

#include <other/core/python/config.h>
#include <other/core/python/exceptions.h>
#include <other/core/utility/config.h>
#ifdef OTHER_PYTHON
#include <other/core/python/wrap_function.h>
#endif
namespace other {
namespace python {

#ifdef OTHER_PYTHON

#ifdef _WIN32
#define MODINIT PyMODINIT_FUNC
#else
#define MODINIT PyMODINIT_FUNC OTHER_EXPORT 
#endif

struct Module {
  OTHER_CORE_EXPORT Module(char const *name);
};

#define OTHER_PYTHON_MODULE(name) \
  static void Init_Helper_##name(); \
  MODINIT init##name(); \
  MODINIT init##name() { \
    try { \
      ::other::python::Module module(#name); \
      ::other::python::import_core(); \
      Init_Helper_##name(); \
    } catch(std::exception& error) { \
      ::other::set_python_exception(error); \
    } \
  } \
  static void Init_Helper_##name()

#else // non-python stub

#define OTHER_PYTHON_MODULE(name) \
  static OTHER_UNUSED void Init_Helper_##name()

#endif

OTHER_CORE_EXPORT void import_core();

// Steal reference to object and add it to the current module
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
#define OTHER_FUNCTION_2(name,f) ::other::python::function(#name,f);

#define OTHER_OVERLOADED_FUNCTION_2(type,name,function_) ::other::python::function(name,(type)function_);

#define OTHER_OVERLOADED_FUNCTION(type,function_) OTHER_OVERLOADED_FUNCTION_2(type,#function_,function_);

struct Scope {
  OTHER_CORE_EXPORT Scope(PyObject* module);
  OTHER_CORE_EXPORT ~Scope();
};

#ifdef OTHER_PYTHON
#define OTHER_WRAP(name) extern void wrap_##name();wrap_##name();
#else
#define OTHER_WRAP(name)
#endif

}
}
