//#####################################################################
// Header module
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
#include <other/core/utility/config.h>
#include <stdexcept>
namespace other {

#ifdef OTHER_PYTHON

OTHER_CORE_EXPORT void module_push(const char* name);
OTHER_CORE_EXPORT void module_pop();

// Defined in core/python/exceptions.cpp, but declared here to minimize includes.
OTHER_CORE_EXPORT void set_python_exception(const std::exception& error);

#ifdef _WIN32
#define OTHER_MODINIT PyMODINIT_FUNC
#define OTHER_EXPORT_HELPER OTHER_EXPORT
#else
#define OTHER_MODINIT PyMODINIT_FUNC OTHER_EXPORT 
#define OTHER_EXPORT_HELPER static
#endif

#define OTHER_PYTHON_MODULE(name) \
  OTHER_EXPORT_HELPER void other_init_helper_##name(); \
  OTHER_MODINIT init##name(); \
  OTHER_MODINIT init##name() { \
    try { \
      ::other::module_push(#name); \
      other_init_helper_##name(); \
    } catch(const std::exception& error) { \
      ::other::set_python_exception(error); \
    } \
    ::other::module_pop(); \
  } \
  void other_init_helper_##name()

#else // non-python stub

#define OTHER_PYTHON_MODULE(name) \
  void other_init_helper_##name()

#endif

#ifndef OTHER_WRAP
#ifdef OTHER_PYTHON
#define OTHER_WRAP(name) extern void wrap_##name();wrap_##name();
#else
#define OTHER_WRAP(name)
#endif
#endif

}
