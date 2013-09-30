// Assertions and other debugging utilities
#pragma once

#include <other/core/utility/config.h>
#include <typeinfo>
#include <string>

// Wow.  The gcc folk have fixed segfault bugs for __FUNCTION__ or __PRETTY_FUNCTION__ numerous times.
// Is this really so complicated?  I am giving up on gcc 4.8 and using 0.
#if defined(__clang__)
#define OTHER_DEBUG_FUNCTION_NAME ((const char*)__PRETTY_FUNCTION__)
#elif defined(__GNUC__) && defined(__GNUC_MINOR__) && __GNUC__==4 && __GNUC_MINOR__==8
#define OTHER_DEBUG_FUNCTION_NAME ("unknown") // gcc 4.8 is broken
#elif defined(__WIN32__)
#define OTHER_DEBUG_FUNCTION_NAME ((const char*)__FUNCSIG__)
#else
#define OTHER_DEBUG_FUNCTION_NAME ((const char*)__FUNCTION__)
#endif

#define OTHER_WARN_IF_NOT_OVERRIDDEN() \
  do{static bool __first_time__=true;if(__first_time__){other::warn_if_not_overridden(OTHER_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,typeid(*this));__first_time__=false;}}while(0)

#define OTHER_WARNING(message) \
  do{static bool __first_time__=true;if(__first_time__){other::warning((message),OTHER_DEBUG_FUNCTION_NAME,__FILE__,__LINE__);__first_time__=false;}}while(0)

#define OTHER_FUNCTION_IS_NOT_DEFINED() \
  other::function_is_not_defined(OTHER_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,typeid(*this))

#define OTHER_NOT_IMPLEMENTED(...) \
  other::not_implemented(OTHER_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,other::debug_message(__VA_ARGS__))

#define OTHER_FATAL_ERROR(...) \
  other::fatal_error(OTHER_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,other::debug_message(__VA_ARGS__))

#define OTHER_ASSERT(condition,...) \
  ((condition) ? (void)0 : other::assertion_failed(OTHER_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,#condition,other::debug_message(__VA_ARGS__)))

#ifdef NDEBUG
#   define OTHER_DEBUG_ONLY(...)
#else
#   define OTHER_DEBUG_ONLY(...) __VA_ARGS__
#endif

namespace other {

using std::string;
using std::type_info;

OTHER_CORE_EXPORT void breakpoint();

// Helper function to work around zero-variadic argument weirdness
static inline const char* debug_message(){return 0;}
static inline const char* debug_message(const char* message){return message;}
static inline const char* debug_message(const string& message){return message.c_str();}

OTHER_CORE_EXPORT void warn_if_not_overridden(const char* function,const char* file,unsigned int line,const type_info& type) OTHER_COLD;
OTHER_CORE_EXPORT void warning(const string& message,const char* function,const char* file,unsigned int line) OTHER_COLD;
OTHER_CORE_EXPORT void OTHER_NORETURN(function_is_not_defined(const char* function,const char* file,unsigned int line,const type_info& type)) OTHER_COLD;
OTHER_CORE_EXPORT void OTHER_NORETURN(not_implemented(const char* function,const char* file,unsigned int line,const char* message)) OTHER_COLD;
OTHER_CORE_EXPORT void OTHER_NORETURN(fatal_error(const char* function,const char* file,unsigned int line,const char* message)) OTHER_COLD;
OTHER_CORE_EXPORT void OTHER_NORETURN(assertion_failed(const char* function,const char* file,unsigned int line,const char* condition,const char* message)) OTHER_COLD;

// Instead of throwing an exception, call the given function when an error occurs
#ifdef _WIN32
typedef void (*ErrorCallback)(const string&);
#else
typedef void OTHER_NORETURN((*ErrorCallback)(const string&));
#endif
OTHER_CORE_EXPORT void set_error_callback(ErrorCallback callback);

}
