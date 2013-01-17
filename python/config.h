// Include Python.h in the appropriate platform specific manner
#pragma once

#ifdef OTHER_PYTHON
#ifdef __APPLE__
#include <Python/Python.h>
#else
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE
#include <Python.h>
#endif
#else
#include <sys/types.h>

#ifdef _WIN32 
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
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

namespace other {

#if OTHER_THREAD_SAFE
#ifdef __GNUC__

// See http://en.wikipedia.org/wiki/Fetch-and-add
template<class T> static inline T fetch_and_add(T* n, T dn) {
  return __sync_fetch_and_add(n,dn);
}

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

#ifdef OTHER_PYTHON

#define OTHER_ONLY_PYTHON(...) __VA_ARGS__
#define OTHER_PY_DEALLOC _Py_Dealloc
#define OTHER_PY_OBJECT_HEAD PyObject_HEAD
#define OTHER_PY_OBJECT_INIT PyObject_INIT

#ifndef _WIN32
struct _object;
struct _typeobject;
namespace other {
typedef _object PyObject;
typedef _typeobject PyTypeObject;
}
#else
namespace other {
using ::PyObject;
using ::PyTypeObject;
}
#endif

#else

#define OTHER_ONLY_PYTHON(...)
#define OTHER_PY_TYPE(op) (((PyObject*)(op))->ob_type)
#define OTHER_PY_DEALLOC(op) ((*OTHER_PY_TYPE(op)->tp_dealloc)((PyObject*)(op)))
#define OTHER_PY_OBJECT_HEAD \
  /* Intentionally reverse the order so that if we accidentally link \
   * non-python aware code with python aware code, we die quickly. */ \
  other::PyTypeObject* ob_type; \
  ssize_t ob_refcnt;
#define OTHER_PY_OBJECT_INIT(op,type) \
  ((((PyObject*)(op))->ob_type=(type)),(((PyObject*)(op))->ob_refcnt=1),(op))

#endif

#if defined(OTHER_PYTHON) && !OTHER_THREAD_SAFE

// Use standard unsafe python reference counting
#define OTHER_INCREF Py_INCREF
#define OTHER_DECREF Py_DECREF
#define OTHER_XINCREF Py_XINCREF
#define OTHER_XDECREF Py_XDECREF

#else

// Use atomics to ensure thread safety in pure C++ code

#define OTHER_INCREF(op) \
  ((void)other::fetch_and_add(&((PyObject*)(op))->ob_refcnt,1l))
#define OTHER_XINCREF(op) do { \
  if (op) OTHER_INCREF(op); } while (false)
#define OTHER_DECREF(op) do { \
  if (other::fetch_and_add(&((PyObject*)(op))->ob_refcnt,-1l)==1)\
    OTHER_PY_DEALLOC(op); } while(false)
#define OTHER_XDECREF(op) do { \
  if (op) OTHER_DECREF(op); } while (false)

#endif

#ifndef OTHER_PYTHON

// Stubs to mimic python
namespace other {

struct PyObject;

struct PyTypeObject {
  const char* tp_name;
  void (*tp_dealloc)(PyObject*);
};

struct PyObject {
  OTHER_PY_OBJECT_HEAD
};

}
#endif
