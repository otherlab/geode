// Buffer: A minimal object containing a buffer of memory.
#pragma once

// Buffer is used by Array<T> to hold memory created from C++.

#include <geode/utility/Object.h>
#include <geode/utility/type_traits.h>
#include <stdlib.h>
namespace geode {

struct Buffer : public Owner {
  // Use 16 byte alignment for SSE purposes.  Should be data[0], but Windows would complain.
  GEODE_ALIGNED(16) char data[1];

  // Make noncopyable and nonconstructible.  To construct, use new_ below.
  Buffer() = delete;
  Buffer(const Buffer&) = delete;
  void operator=(const Buffer&) = delete;

  template<class T> static shared_ptr<Buffer> new_(const int m) {
#ifndef _WIN32
    static_assert(has_trivial_destructor<T>::value,"Array<T> never calls destructors, so T cannot have any");
    return shared_ptr<Buffer>((Buffer*)malloc(16+m*sizeof(T)));
#else
    // Windows doesn't guarantee 16 byte alignment, so use _aligned_malloc
    return shared_ptr<Buffer>((Buffer*)_aligned_malloc(16+m*sizeof(T),16));
#endif
  }
};

// Check alignment constraints
static_assert(offsetof(Buffer,data)==16,"data must be 16 byte aligned for SSE purposes");

}
