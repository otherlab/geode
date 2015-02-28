// Include Python.h in the appropriate platform specific manner
#pragma once

#include <geode/config.h>

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
    #endif
  #endif
#endif

#if defined(_WIN32) && GEODE_THREAD_SAFE
// Needed for _InterlockedExchangeAdd
#include <intrin.h>
#endif

namespace geode {

#if GEODE_THREAD_SAFE
#if defined(__GNUC__)
// See http://en.wikipedia.org/wiki/Fetch-and-add
template<class T> static inline T fetch_and_add(T* n, T dn) {
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

}

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
  ((void)geode::fetch_and_add(&((PyObject*)(op))->ob_refcnt,(ssize_t)1l))
#define GEODE_XINCREF(op) do { \
  if (op) GEODE_INCREF(op); } while (false)
#define GEODE_DECREF(op) do { \
  if (geode::fetch_and_add(&((PyObject*)(op))->ob_refcnt,(ssize_t)-1l)==1)\
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
