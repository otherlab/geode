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
#include <boost/static_assert.hpp>
#include <boost/type_traits/has_trivial_destructor.hpp>
#include <stdlib.h>
namespace other {

namespace mpl = boost::mpl;

struct Buffer {
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
  OTHER_PY_OBJECT_HEAD // Contains a reference count and a pointer to the type object
  OTHER_ALIGNED(16) char data[1]; // Use 16 byte alignment for SSE purposes.  Should be data[0], but Windows would complain.

private:
  Buffer(); // Should never be called
  Buffer(const Buffer&);
  void operator=(const Buffer&);
public:

  template<class T> static Buffer*
  new_(const int m) {
#ifndef _WIN32
    BOOST_MPL_ASSERT((boost::has_trivial_destructor<T>)); // Array<T> never calls destructors, so T cannot have any
    Buffer* self = (Buffer*)malloc(16+m*sizeof(T));
#else
    // Windows doesn't guarantee 16 byte alignment, so use _aligned_malloc
    Buffer* self = (Buffer*)_aligned_malloc(16+m*sizeof(T),16);
    //if (m > 100000) {
    //    char str[2000];
    //    sprintf(str, "Buffer of length %d and size %d\n", m, sizeof(T));
    //    OutputDebugStringA(str);
    //}
#endif
    return OTHER_PY_OBJECT_INIT(self,&pytype);
  }
};

// Check alignment constraints
BOOST_STATIC_ASSERT(offsetof(Buffer,data)==16);

}
