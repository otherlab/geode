// Macro to stack allocate a RawArray
#pragma once

#include <geode/array/RawArray.h>
#ifdef _WIN32
#include <malloc.h>
#define GEODE_ALLOCA _alloca
#else
#include <alloca.h>
#define GEODE_ALLOCA alloca
#endif

// GEODE_RAW_ALLOCA(count,Type) stack allocates an uninitialized buffer using alloca, and returns it as a RawArray.
// IMPORTANT: On Windows, the size argument is evaluated twice, since there is no equivalent to gcc's
// statement expressions.  If the count argument must be evaluated once, *do not* use this macro.
#ifdef __GNUC__
#define GEODE_RAW_ALLOCA(count,...) ({ \
  const int _count = (count); \
  ::geode::RawArray<__VA_ARGS__>(_count,(__VA_ARGS__*)GEODE_ALLOCA(sizeof(__VA_ARGS__)*_count)); })
#else // Fallback version that evaluates the count argument twice
#define GEODE_RAW_ALLOCA(count,...) \
  (::geode::RawArray<__VA_ARGS__>((count),(__VA_ARGS__*)GEODE_ALLOCA(sizeof(__VA_ARGS__)*(count))))
#endif
