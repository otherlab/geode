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

template<class T,class... Args> static inline Ref<T> new_(Args&&... args) {
  // Note that we can't go through tp_alloc, since the actual object size will be larger than tp_basicsize
  // if we have a C++ class that's derived from a Python type but isn't itself exposed to Python.
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

}
#include <other/core/python/Ref.h>
