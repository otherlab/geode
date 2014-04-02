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

#include <geode/python/config.h>
#include <geode/utility/config.h>
#include <stdexcept>
namespace geode {

#ifdef GEODE_PYTHON

GEODE_CORE_EXPORT void module_push(const char* name);
GEODE_CORE_EXPORT void module_pop();

// Defined in geode/python/exceptions.cpp, but declared here to minimize includes.
GEODE_CORE_EXPORT void set_python_exception(const std::exception& error);

#ifdef _WIN32
#define GEODE_MODINIT PyMODINIT_FUNC
#define GEODE_EXPORT_HELPER GEODE_EXPORT
#else
#define GEODE_MODINIT PyMODINIT_FUNC GEODE_EXPORT 
#define GEODE_EXPORT_HELPER static
#endif

#define GEODE_PYTHON_MODULE(name) \
  GEODE_EXPORT_HELPER void geode_init_helper_##name(); \
  GEODE_MODINIT init##name(); \
  GEODE_MODINIT init##name() { \
    try { \
      ::geode::module_push(#name); \
      geode_init_helper_##name(); \
    } catch(const std::exception& error) { \
      ::geode::set_python_exception(error); \
    } \
    ::geode::module_pop(); \
  } \
  void geode_init_helper_##name()

#else // non-python stub

#define GEODE_PYTHON_MODULE(name) \
  void geode_init_helper_##name()

#endif

#ifndef GEODE_WRAP
#ifdef GEODE_PYTHON
#define GEODE_WRAP(name) extern void wrap_##name();wrap_##name();
#else
#define GEODE_WRAP(name)
#endif
#endif

}
