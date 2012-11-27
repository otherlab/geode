//#####################################################################
// Function new_
//#####################################################################
//
// new_<T>(args...) creates a new mixed python/C++ object via T(args...), and returns a reference to it (as Ref<T>).
//
// T must be both a C++ class and a python object type, and must expose a static pytype member.  See Python/Object.h for how to arrange this for new classes.
//
//#####################################################################
#pragma once

#include <other/core/python/forward.h>
#include <other/core/math/max.h>
#include <new>
namespace other {

#define PASS(...) __VA_ARGS__

#ifdef OTHER_VARIADIC

template<class T,class... Args> static inline Ref<T> new_(Args&&... args) {
  /* Note that we can't go through tp_alloc, since the actual object size will be larger than tp_basicsize
   * if we have a C++ class that's derived from a Python type but isn't itself exposed to Python. */
  PyObject* memory = (PyObject*)malloc(sizeof(PyObject)+sizeof(T));
  if (!memory) throw std::bad_alloc();
  memory = PyObject_INIT(memory,&T::pytype);
  try {
    new(memory+1) T(args...);
    Ref<T> ref;
    ref.self = (T*)(memory+1);
    ref.owner_ = memory;
    return ref;
  } catch(...) {
    free(memory);
    throw;
  }
}

#else // Unpleasant nonvariadic versions

#define OTHER_DEFINE_NEW(ARGS,Args,args) \
  OTHER_DEFINE_NEW_2((,OTHER_REMOVE_PARENS(ARGS)),Args,args)

#define OTHER_DEFINE_NEW_2(CARGS,Args,args) \
  template<class T OTHER_REMOVE_PARENS(CARGS)> static inline Ref<T> new_ Args { \
    /* Note that we can't go through tp_alloc, since the actual object size will be larger than tp_basicsize
     * if we have a C++ class that's derived from a Python type but isn't itself exposed to Python. */ \
    PyObject* memory = (PyObject*)malloc(sizeof(PyObject)+sizeof(T)); \
    if (!memory) throw std::bad_alloc(); \
    memory = PyObject_INIT(memory,&T::pytype); \
    try { \
      new(memory+1) T args; \
      Ref<T> ref; \
      ref.self = (T*)(memory+1); \
      ref.owner_ = memory; \
      return ref; \
    } catch(...) { \
      free(memory); \
      throw; \
    } \
  }

OTHER_DEFINE_NEW_2((),(),())
OTHER_DEFINE_NEW((class A0),(A0&& a0),(a0))
OTHER_DEFINE_NEW((class A0,class A1),(A0&& a0,A1&& a1),(a0,a1))
OTHER_DEFINE_NEW((class A0,class A1,class A2),(A0&& a0,A1&& a1,A2&& a2),(a0,a1,a2))
OTHER_DEFINE_NEW((class A0,class A1,class A2,class A3),(A0&& a0,A1&& a1,A2&& a2,A3&& a3),(a0,a1,a2,a3))
OTHER_DEFINE_NEW((class A0,class A1,class A2,class A3,class A4),(A0&& a0,A1&& a1,A2&& a2,A3&& a3,A4&& a4),(a0,a1,a2,a3,a4))
OTHER_DEFINE_NEW((class A0,class A1,class A2,class A3,class A4,class A5),(A0&& a0,A1&& a1,A2&& a2,A3&& a3,A4&& a4,A5&& a5),(a0,a1,a2,a3,a4,a5))

#endif

#undef OTHER_DEFINE_NEW

}
#include <other/core/python/Ref.h>
