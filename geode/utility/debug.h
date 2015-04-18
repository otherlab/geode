// Assertions and other debugging utilities
#pragma once

#include <geode/utility/config.h>
#include <typeinfo>
#include <string>

// Wow.  The gcc folk have fixed segfault bugs for __FUNCTION__ or __PRETTY_FUNCTION__ numerous times.
// Is this really so complicated?  I am giving up on gcc 4.8 and using 0.
#if defined(__clang__) || defined(__MINGW32__)
#define GEODE_DEBUG_FUNCTION_NAME ((const char*)__PRETTY_FUNCTION__)
#elif defined(__GNUC__) && defined(__GNUC_MINOR__) && __GNUC__==4 && __GNUC_MINOR__==8
#define GEODE_DEBUG_FUNCTION_NAME ("unknown") // gcc 4.8 is broken
#elif defined(__WIN32__)
#define GEODE_DEBUG_FUNCTION_NAME ((const char*)__FUNCSIG__)
#else
#define GEODE_DEBUG_FUNCTION_NAME ((const char*)__FUNCTION__)
#endif

#define GEODE_WARN_IF_NOT_OVERRIDDEN() \
  do{static bool __first_time__=true;if(__first_time__){geode::warn_if_not_overridden(GEODE_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,typeid(*this));__first_time__=false;}}while(0)

#define GEODE_WARNING(message) \
  do{static bool __first_time__=true;if(__first_time__){geode::warning((message),GEODE_DEBUG_FUNCTION_NAME,__FILE__,__LINE__);__first_time__=false;}}while(0)

#define GEODE_FUNCTION_IS_NOT_DEFINED() \
  geode::function_is_not_defined(GEODE_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,typeid(*this))

#define GEODE_NOT_IMPLEMENTED(...) \
  geode::not_implemented(GEODE_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,geode::debug_message(__VA_ARGS__))

#define GEODE_FATAL_ERROR(...) \
  geode::fatal_error(GEODE_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,geode::debug_message(__VA_ARGS__))

#define GEODE_ASSERT(condition,...) \
  ((condition) ? (void)0 : geode::assertion_failed(GEODE_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,#condition,geode::debug_message(__VA_ARGS__)))

#ifdef NDEBUG
#   define GEODE_DEBUG_ONLY(...)
#else
#   define GEODE_DEBUG_ONLY(...) __VA_ARGS__
#endif

namespace geode {

using std::string;
using std::type_info;

GEODE_CORE_EXPORT void breakpoint();

// Helper function to work around zero-variadic argument weirdness
static inline const char* debug_message(){return 0;}
static inline const char* debug_message(const char* message){return message;}
static inline const char* debug_message(const string& message){return message.c_str();}

GEODE_CORE_EXPORT void warn_if_not_overridden(const char* function,const char* file,unsigned int line,const type_info& type) GEODE_COLD;
GEODE_CORE_EXPORT void warning(const string& message,const char* function,const char* file,unsigned int line) GEODE_COLD;
GEODE_CORE_EXPORT void GEODE_NORETURN(function_is_not_defined(const char* function,const char* file,unsigned int line,const type_info& type)) GEODE_COLD;
GEODE_CORE_EXPORT void GEODE_NORETURN(not_implemented(const char* function,const char* file,unsigned int line,const char* message)) GEODE_COLD;
GEODE_CORE_EXPORT void GEODE_NORETURN(fatal_error(const char* function,const char* file,unsigned int line,const char* message)) GEODE_COLD;
GEODE_CORE_EXPORT void GEODE_NORETURN(assertion_failed(const char* function,const char* file,unsigned int line,const char* condition,const char* message)) GEODE_COLD;
GEODE_CORE_EXPORT void GEODE_NORETURN(did_not_raise(const type_info& error, const char* message)) GEODE_COLD;

// Instead of throwing an exception, call the given function when an error occurs
#ifdef _WIN32
typedef void (*ErrorCallback)(const string&);
#else
typedef void GEODE_NORETURN((*ErrorCallback)(const string&));
#endif
GEODE_CORE_EXPORT void set_error_callback(ErrorCallback callback);

// Assert that a function raises an exception
template<class Error,class F,class... M> static void assert_raises(const F& f, const M&... message) {
  try {
    f();
  } catch (const Error&) {
    return;
  }
  did_not_raise(typeid(Error),debug_message(message...));
}

}
