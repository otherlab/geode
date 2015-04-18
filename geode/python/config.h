// Include Python.h in the appropriate platform specific manner
#pragma once
#include <geode/config.h>

#ifdef __MINGW32__
// MinGW attempts to define this unconditionally, but not always early enough to keep things in sync
// Define it here so things don't break later
#undef __USE_MINGW_ANSI_STDIO
#define __USE_MINGW_ANSI_STDIO 1
#endif

#ifdef GEODE_PYTHON
  #ifdef __APPLE__
    #include <Python.h>
    // Clean up macros in anticipation of C++ headers
    #undef isspace
    #undef isupper
    #undef islower
    #undef isalpha
    #undef isalnum
    #undef toupper
    #undef tolower
  #else // Not __APPLE__
    #undef _POSIX_C_SOURCE
    #undef _XOPEN_SOURCE

    #if defined(__MINGW32__)
      // Python.h compiled with MinGW, defines hypot as _hypot. This then gets undefined in the cmath header, but only after breaking the declaration
      // This is a known issue: bugs.python.org/issue11566
      // One workaround would be to include cmath before Python.h, but Python.h some of the preprocessor defines are supposed to alter system headers
      // An alternative workaround is to #define _hypot as hypot which thwarts Python's attempt to rename it
      // Both of these options seem dangerous, but I think any problems caused by this option would be easier to debug:
      #define _hypot hypot
    #endif

    #if defined(__MINGW64__)
      // Python doesn't detect 64 bit compilation except with MSC
      // See: bugs.python.org/issue4709
      // We force that define here
      #define MS_WIN64
    #endif

    #if defined(_WIN32) && defined(_DEBUG)
      // Extensions targeting debug flavor of python interpreter aren't usable with release versions
      // Instead of requiring installation of separate interpreter for debugging we attempt to build and link against the release interpreter
      // http://www.boost.org/doc/libs/1_57_0/libs/python/doc/building.html#python-debugging-builds
      // This behavior could be made configurable if needed
      #pragma push_macro("_DEBUG")
      #undef _DEBUG
      #include <Python.h>
      #pragma pop_macro("_DEBUG")
    #else
      #include <Python.h>
    #endif
  #endif

  #if defined(__MINGW32__)
    // The 'format' attribute of PyErr_Format is 'printf' which on windows causes an "unknown conversion type character" error for things like "%zd"
    // The correct value should be 'gnu_printf' since python implements its own formatting that always accepts gnu style formatting characters
    // As a workaround we cast a reference to PyErr_Format to the same type except using the 'gnu_printf' attribute
    // I'm not sure if this is intended to work, but I see no complaints from the compiler (except helpful warnings when formatting characters are wrong)
    // Although this is ugly it helped catch several formatting issues that would have been missed if formatting warnings were simply disabled
    // PyErr_Format can still be called directly if the format string would be compatible with ms_printf and gnu_printf
    #define GEODE_CALL_PyErr_Format(...) static_cast<PyObject*(__attribute__((__format__(gnu_printf,2,3)))&)(PyObject*,const char*,...)>(PyErr_Format)(__VA_ARGS__)
  #else
    #define GEODE_CALL_PyErr_Format(...) PyErr_Format(__VA_ARGS__)
  #endif

  // We have to manually define MS_WIN64 in geode/python/config.h when using MinGW to work around configuration bug in Python.h
  // We check that this worked here
  #if defined(MS_WIN64) != defined(_WIN64)
    #error "configuration of MS_WIN64 doesn't match _WIN64"
  #endif

#else // Not GEODE_PYTHON
  #include <stdint.h>
  #include <sys/types.h>

  #ifdef _WIN32
    #include <windows.h>
    #include <stdint.h>
    typedef SSIZE_T ssize_t;
    #ifndef LEAVE_WINDOWS_DEFINES_ALONE
      // clean up after windows.h
      #undef min
      #undef max
      #undef far
      #undef near
      #undef interface
      #undef small
      #ifdef __MINGW32__
        #undef NEAR
      #endif
    #endif
  #endif
#endif

#if GEODE_THREAD_SAFE && !defined(__GNUC__)
// Needed for _InterlockedExchangeAdd
#include <intrin.h>
#endif

namespace geode {

#if GEODE_THREAD_SAFE
#if defined(__GNUC__)
// See http://en.wikipedia.org/wiki/Fetch-and-add
template<class T> static inline T fetch_and_add(volatile T* n, T dn) {
  return __sync_fetch_and_add(n,dn);
}
#elif defined(_WIN32)
template<class T> static inline T fetch_and_add(volatile T* n, T dn);
template<> static inline long fetch_and_add<long>(volatile long* n, long dn) { return _InterlockedExchangeAdd(n,dn); }
template<> static inline __int64 fetch_and_add<__int64>(volatile __int64* n, __int64 dn) { return _InterlockedExchangeAdd64(n,dn); }
#else
#error "Don't know atomic fetch and add for this compiler"
#endif

#else // non-threadsafe

template<class T> static inline T fetch_and_add(T* n, T dn) {
  const T old = *n;
  *n += dn;
  return old;
}

#endif

// Workaround for type deduction issues
template<class T> static inline T fetch_and_add_i(volatile T* n, int dn) { return fetch_and_add<T>(n,(T)dn); }

} // end namespace geode

#ifdef GEODE_PYTHON

#define GEODE_ONLY_PYTHON(...) __VA_ARGS__
#define GEODE_PY_DEALLOC _Py_Dealloc
#define GEODE_PY_OBJECT_HEAD PyObject_HEAD
#define GEODE_PY_OBJECT_INIT PyObject_INIT

#ifndef _WIN32
struct _object;
struct _typeobject;
namespace geode {
typedef _object PyObject;
typedef _typeobject PyTypeObject;
}
#else
namespace geode {
using ::PyObject;
using ::PyTypeObject;
}
#endif

#else

#define GEODE_ONLY_PYTHON(...)
#define GEODE_PY_TYPE(op) (((PyObject*)(op))->ob_type)
#define GEODE_PY_DEALLOC(op) ((*GEODE_PY_TYPE(op)->tp_dealloc)((PyObject*)(op)))
#define GEODE_PY_OBJECT_HEAD \
  /* Intentionally reverse the order so that if we accidentally link \
   * non-python aware code with python aware code, we die quickly. */ \
  geode::PyTypeObject* ob_type; \
  ssize_t ob_refcnt;
#define GEODE_PY_OBJECT_INIT(op,type) \
  ((((PyObject*)(op))->ob_type=(type)),(((PyObject*)(op))->ob_refcnt=1),(op))

#endif

#if defined(GEODE_PYTHON) && !GEODE_THREAD_SAFE

// Use standard unsafe python reference counting
#define GEODE_INCREF Py_INCREF
#define GEODE_DECREF Py_DECREF
#define GEODE_XINCREF Py_XINCREF
#define GEODE_XDECREF Py_XDECREF

#else

// Use atomics to ensure thread safety in pure C++ code

#define GEODE_INCREF(op) \
  ((void)geode::fetch_and_add_i(&((PyObject*)(op))->ob_refcnt,1))
#define GEODE_XINCREF(op) do { \
  if (op) GEODE_INCREF(op); } while (false)
#define GEODE_DECREF(op) do { \
  if (geode::fetch_and_add_i(&((PyObject*)(op))->ob_refcnt,-1)==1)\
    GEODE_PY_DEALLOC(op); } while(false)
#define GEODE_XDECREF(op) do { \
  if (op) GEODE_DECREF(op); } while (false)

#endif

#ifndef GEODE_PYTHON

// Stubs to mimic python
namespace geode {

struct PyObject;

struct PyTypeObject {
  const char* tp_name;
  void (*tp_dealloc)(PyObject*);
};

struct PyObject {
  GEODE_PY_OBJECT_HEAD
};

typedef intptr_t Py_intptr_t;

}
#endif

// This should get moved to geode/utility/config.h once we ensure __USE_MINGW_ANSI_STDIO gets defined first
#if defined(__USE_MINGW_ANSI_STDIO) && ((__USE_MINGW_ANSI_STDIO + 0) != 0)
// It appears that we need to use gnu_printf even if we have __USE_MINGW_ANSI_STDIO 
#define GEODE_FORMAT_PRINTF(fmt,list) GEODE_FORMAT(gnu_printf,fmt,list)
#else
#define GEODE_FORMAT_PRINTF(fmt,list) GEODE_FORMAT(printf,fmt,list)
#endif
