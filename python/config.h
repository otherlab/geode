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

// See http://en.wikipedia.org/wiki/Fetch-and-add
static inline int fetch_and_add(int* n, int dn) {
  asm volatile( 
    "lock; xaddl %%eax, %2;"
    : "=a" (dn) // output
    : "a" (dn), "m" (*n) // input
    :"memory");
  // Return the old value of n
  return dn;
}

#else // non-threadsafe

static inline int fetch_and_add(int* n, int dn) {
  const int old = *n;
  *n += dn;
  return old;
}

#endif

// Treat *n as little endian and assume everything beyond 32 bits is zero
static inline int hack_fetch_and_add(ssize_t* n, int dn) {
  return fetch_and_add((int*)n,dn);
}

}

#ifdef OTHER_PYTHON

#define OTHER_ONLY_PYTHON(...) __VA_ARGS__
#define OTHER_PY_DEALLOC _Py_Dealloc
#define OTHER_PY_OBJECT_HEAD PyObject_HEAD
#define OTHER_PY_OBJECT_INIT PyObject_INIT
struct _object;
struct _typeobject;
namespace other {
typedef _object PyObject;
typedef _typeobject PyTypeObject;
};

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
  ((void)other::hack_fetch_and_add(&((PyObject*)(op))->ob_refcnt,1))
#define OTHER_XINCREF(op) do { \
  if (op) OTHER_INCREF(op); } while (false)
#define OTHER_DECREF(op) do { \
  if (other::hack_fetch_and_add(&((PyObject*)(op))->ob_refcnt,-1)==1)\
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
