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

#include <geode/utility/forward.h>
#include <geode/math/max.h>
#include <new>
namespace geode {

#ifdef GEODE_VARIADIC

template<class T,class... Args> static inline Ref<T> new_(Args&&... args) {
  /* Note that we can't go through tp_alloc, since the actual object size will be larger than tp_basicsize
   * if we have a C++ class that's derived from a Python type but isn't itself exposed to Python. */
  geode::PyObject* memory = (geode::PyObject*)malloc(sizeof(geode::PyObject)+sizeof(T));
  if (!memory) throw std::bad_alloc();
  memory = GEODE_PY_OBJECT_INIT(memory,&T::pytype);
  try {
    new(memory+1) T(args...);
    return Ref<T>(*(T*)(memory+1),memory,typename Ref<T>::Steal());
  } catch(...) {
    free(memory);
    throw;
  }
}

#else // Unpleasant nonvariadic versions

#define GEODE_DEFINE_NEW(ARGS,Args,args) \
  GEODE_DEFINE_NEW_2((,GEODE_REMOVE_PARENS(ARGS)),Args,args)

#define GEODE_DEFINE_NEW_2(CARGS,Args,args) \
  template<class T GEODE_REMOVE_PARENS(CARGS)> static inline Ref<T> new_ Args { \
    /* Note that we can't go through tp_alloc, since the actual object size will be larger than tp_basicsize
     * if we have a C++ class that's derived from a Python type but isn't itself exposed to Python. */ \
    geode::PyObject* memory = (geode::PyObject*)malloc(sizeof(geode::PyObject)+sizeof(T)); \
    if (!memory) throw std::bad_alloc(); \
    memory = GEODE_PY_OBJECT_INIT(memory,&T::pytype); \
    try { \
      new(memory+1) T args; \
      return Ref<T>(*(T*)(memory+1),memory,typename Ref<T>::Steal()); \
    } catch(...) { \
      free(memory); \
      throw; \
    } \
  }

GEODE_DEFINE_NEW_2((),(),())
GEODE_DEFINE_NEW((class A0),(A0&& a0),(a0))
GEODE_DEFINE_NEW((class A0,class A1),(A0&& a0,A1&& a1),(a0,a1))
GEODE_DEFINE_NEW((class A0,class A1,class A2),(A0&& a0,A1&& a1,A2&& a2),(a0,a1,a2))
GEODE_DEFINE_NEW((class A0,class A1,class A2,class A3),(A0&& a0,A1&& a1,A2&& a2,A3&& a3),(a0,a1,a2,a3))
GEODE_DEFINE_NEW((class A0,class A1,class A2,class A3,class A4),(A0&& a0,A1&& a1,A2&& a2,A3&& a3,A4&& a4),(a0,a1,a2,a3,a4))
GEODE_DEFINE_NEW((class A0,class A1,class A2,class A3,class A4,class A5),(A0&& a0,A1&& a1,A2&& a2,A3&& a3,A4&& a4,A5&& a5),(a0,a1,a2,a3,a4,a5))
GEODE_DEFINE_NEW((class A0,class A1,class A2,class A3,class A4,class A5,class A6),(A0&& a0,A1&& a1,A2&& a2,A3&& a3,A4&& a4,A5&& a5,A6&& a6),(a0,a1,a2,a3,a4,a5,a6))
GEODE_DEFINE_NEW((class A0,class A1,class A2,class A3,class A4,class A5,class A6,class A7),(A0&& a0,A1&& a1,A2&& a2,A3&& a3,A4&& a4,A5&& a5,A6&& a6,A7&& a7),(a0,a1,a2,a3,a4,a5,a6,a7))
GEODE_DEFINE_NEW((class A0,class A1,class A2,class A3,class A4,class A5,class A6,class A7,class A8),(A0&& a0,A1&& a1,A2&& a2,A3&& a3,A4&& a4,A5&& a5,A6&& a6,A7&& a7,A8&& a8),(a0,a1,a2,a3,a4,a5,a6,a7,a8))

#endif

#undef GEODE_DEFINE_NEW

}
#include <geode/python/Ref.h>
