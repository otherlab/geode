//#####################################################################
// Function format
//#####################################################################
#include <other/core/utility/format.h>
#include <cstdarg>
#include <cstdio>
namespace other {

std::string format_helper(const char* format,...) {
  va_list marker;va_start(marker,format);
  static char buffer[2048];
  vsnprintf(buffer,sizeof(buffer)-1,format,marker);
  va_end(marker);
  return buffer;
}

}
