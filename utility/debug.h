//#####################################################################
// Namespace debug
//#####################################################################
#pragma once

#include <other/core/utility/config.h>
#include <typeinfo>
#include <string>

#ifdef _WIN32
#define OTHER_DEBUG_FUNCTION_NAME ((const char*)__FUNCTION__) // cast to const char* to work around error in noreturn
#else
#define OTHER_DEBUG_FUNCTION_NAME ((const char*)__PRETTY_FUNCTION__) // cast to const char* to work around error in noreturn
#endif

#define OTHER_WARN_IF_NOT_OVERRIDDEN() \
  do{static bool __first_time__=true;if(__first_time__){other::debug::warn_if_not_overridden(OTHER_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,typeid(*this));__first_time__=false;}}while(0)

#define OTHER_WARNING(message) \
  do{static bool __first_time__=true;if(__first_time__){other::debug::warning((message),OTHER_DEBUG_FUNCTION_NAME,__FILE__,__LINE__);__first_time__=false;}}while(0)

#define OTHER_FUNCTION_IS_NOT_DEFINED() \
  other::debug::function_is_not_defined(OTHER_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,typeid(*this))

#define OTHER_NOT_IMPLEMENTED(...) \
  other::debug::not_implemented(OTHER_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,other::debug::message(__VA_ARGS__))

#define OTHER_FATAL_ERROR(...) \
  other::debug::fatal_error(OTHER_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,other::debug::message(__VA_ARGS__))

#ifdef __GNUC__ // Avoid ambiguous else warnings on gcc
#define OTHER_ASSERT(condition,...) ({ \
  if(!(condition)) other::debug::assertion_failed(OTHER_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,#condition,other::debug::message(__VA_ARGS__)); })
#else
#define OTHER_ASSERT(condition,...) \
  if(condition){}else{other::debug::assertion_failed(OTHER_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,#condition,other::debug::message(__VA_ARGS__));}
#endif

#ifdef NDEBUG
#   define OTHER_DEBUG_ONLY(...)
#else
#   define OTHER_DEBUG_ONLY(...) __VA_ARGS__
#endif

namespace other {

using std::string;
using std::type_info;

OTHER_CORE_EXPORT void breakpoint();

namespace debug {

// Helper function to work around zero-variadic argument weirdness
static inline const char* message(){return 0;}
static inline const char* message(const char* message){return message;}
static inline const char* message(const string& message){return message.c_str();}

OTHER_CORE_EXPORT void warn_if_not_overridden(const char* function,const char* file,unsigned int line,const type_info& type);
OTHER_CORE_EXPORT void warning(const string& message,const char* function,const char* file,unsigned int line);
OTHER_CORE_EXPORT void OTHER_NORETURN(function_is_not_defined(const char* function,const char* file,unsigned int line,const type_info& type));
OTHER_CORE_EXPORT void OTHER_NORETURN(not_implemented(const char* function,const char* file,unsigned int line,const char* message));
OTHER_CORE_EXPORT void OTHER_NORETURN(fatal_error(const char* function,const char* file,unsigned int line,const char* message));
OTHER_CORE_EXPORT void OTHER_NORETURN(assertion_failed(const char* function,const char* file,unsigned int line,const char* condition,const char* message));

// Instead of throwing an exception, call the given function when an error occurs
#ifdef _WIN32
typedef void (*ErrorCallback)(const string&);
#else
typedef void OTHER_NORETURN((*ErrorCallback)(const string&));
#endif
OTHER_CORE_EXPORT void set_error_callback(ErrorCallback callback);

}
}
