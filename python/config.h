// Include Python.h in the appropriate platform specific manner
#pragma once

#ifdef __APPLE__
#include <Python/Python.h>
#else
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE
#include <Python.h>
#endif

#if !OTHER_THREAD_SAFE

// Use standard unsafe python reference counting
#define OTHER_INCREF Py_INCREF
#define OTHER_DECREF Py_DECREF
#define OTHER_XINCREF Py_XINCREF
#define OTHER_XDECREF Py_XDECREF

#else

// Use atomics to ensure thread safety in pure C++ code

namespace other {

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

// Treat *n as little endian and assume everything beyond 32 bits is zero
static inline int hack_fetch_and_add(Py_ssize_t* n, int dn) {
  return fetch_and_add((int*)n,dn);
}

}

#define OTHER_INCREF(op) \
  ((void)other::hack_fetch_and_add(&((PyObject*)(op))->ob_refcnt,1))
#define OTHER_XINCREF(op) ({ \
  if (op) OTHER_INCREF(op); })
#define OTHER_DECREF(op) ({ \
  if (other::hack_fetch_and_add(&((PyObject*)(op))->ob_refcnt,-1)==1) \
    _Py_Dealloc((PyObject*)(op)); })
#define OTHER_XDECREF(op) ({ \
  if (op) OTHER_DECREF(op); })
  
#endif
