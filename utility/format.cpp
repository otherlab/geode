//#####################################################################
// Function format
//#####################################################################
#include <other/core/utility/format.h>
#include <cstdarg>
#include <cstdio>


// Windows silliness
#undef small
#undef far
#undef near

namespace other {

string format_helper(const char* format,...) {
  // Try using a small buffer first
  va_list marker;
  va_start(marker,format);
  char small[64];
  int n = vsnprintf(small,sizeof(small)-1,format,marker);
  va_end(marker);
  if (unsigned(n) < sizeof(small))
    return small;

#ifdef _WIN32
  // On Windows, vsnprintf returns a useless negative number on failure,
  // we need to call a separate function to get the correct length.
  va_start(marker,format);
  n = _vscprintf(format,marker);
  va_end(marker);
#endif

  // Retry using the exact buffer size
  va_start(marker,format);
  string large(n,'\0');
  vsnprintf(&large[0],n+1,format,marker);
  va_end(marker);
  return large;
}

}
