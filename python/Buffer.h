//#####################################################################
// Class Buffer
//#####################################################################
//
// This is a minimal python object containing a buffer of memory.
// Since it's a python object, it has a reference count, and can be shared.
// No destructors are called, so has_trivial_destructor<T> must be true.
//
//#####################################################################
#pragma once

#include <other/core/python/config.h>
#include <other/core/python/forward.h>
#include <other/core/utility/config.h>
#include <boost/mpl/assert.hpp>
#include <boost/type_traits/has_trivial_destructor.hpp>
#include <stdlib.h>
namespace other {

namespace mpl = boost::mpl;

struct Buffer {
  OTHER_DECLARE_TYPE
  OTHER_PY_OBJECT_HEAD // contains a reference count and a pointer to the type object
  char data[1]; // should be size zero, but Windows would complain

private:
  Buffer(); // should never be called
  Buffer(const Buffer&);
  void operator=(const Buffer&);
public:

  template<class T> static Buffer*
  new_(const int m) {
#ifndef _WIN32
    BOOST_MPL_ASSERT((boost::has_trivial_destructor<T>));
#endif
    Buffer* self = (Buffer*)malloc(sizeof(PyObject)+m*sizeof(T));
    return OTHER_PY_OBJECT_INIT(self,&pytype);
  }
};
}
