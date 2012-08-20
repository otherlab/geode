//#####################################################################
// Macro OTHER_DEBUG_PRINT
//#####################################################################
#include <other/core/utility/DebugPrint.h>
#include <other/core/utility/Log.h>
#include <cstdarg>
namespace other {

void debug_print_helper(const char* prefix,...) {
  Log::cerr<<prefix<<": ";
  va_list marker;va_start(marker,prefix);
  bool first=true;
  for (;;) {
    char* name = va_arg(marker,char*);if(!name) break;
    char* value = va_arg(marker,char*);if(!value) break;
    if (!first) Log::cerr<<", ";first=false;
    Log::cerr<<name<<"="<<value;
  }
  va_end(marker);
  Log::cerr<<std::endl;
}

void debug_print_helper_multiline(const char* prefix,...) {
  Log::cerr<<prefix<<": "<<std::endl;
  va_list marker;va_start(marker,prefix);
  for (;;) {
    char* name = va_arg(marker,char*);if(!name) break;
    char* value = va_arg(marker,char*);if(!value) break;
    Log::cerr<<"    "<<name<<"="<<value<<std::endl;
  }
  va_end(marker);
}

void debug_print_single(const char* prefix) {
  Log::cerr<<prefix<<std::endl;
}

void debug_print_msg(const char* prefix, const char* msg) {
  Log::cerr<<prefix<<msg<<std::endl;
}

}
