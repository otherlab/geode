// Macro to stack allocate a RawArray
#pragma once

#include <other/core/array/RawArray.h>
#include <alloca.h>

// OTHER_RAW_ALLOCA(count,Type) stack allocates an uninitialized buffer using alloca, and returns it as a RawArray.
// IMPORTANT: On Windows, the size argument is evaluated twice, since there is no equivalent to gcc's
// statement expressions.  If the count argument must be evaluated once, *do not* use this macro.
#ifdef __GNUC__
#define OTHER_RAW_ALLOCA(count,...) ({ \
  const int _count = (count); \
  ::other::RawArray<__VA_ARGS__>(_count,(__VA_ARGS__*)alloca(sizeof(__VA_ARGS__)*_count)); })
#else // Fallback version that evaluates the count argument twice
#define OTHER_RAW_ALLOCA(count,...) \
  (::other::RawArray<__VA_ARGS__>((count),(__VA_ARGS__*)alloca(sizeof(__VA_ARGS__)*(count))))
#endif
