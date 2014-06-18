// Assertions and other debugging utilities
#include <geode/python/exceptions.h>
#include <geode/utility/debug.h>
#include <geode/utility/format.h>
#include <geode/utility/process.h>
#include <geode/utility/Log.h>
#include <cassert>
#include <stdexcept>
#if defined(__linux__) || defined(__CYGWIN__) || defined(__APPLE__)
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>
#endif
namespace geode {

using std::endl;
using std::flush;

#if defined(__linux__) || defined(__CYGWIN__) || defined(__APPLE__)
#define BREAKPOINT() raise(SIGINT)
#else
#define BREAKPOINT() assert(false)
#endif

void breakpoint() {
  BREAKPOINT(); // if you use this you need to step out of the signal handler to get a non-corrupt stack
}

static ErrorCallback error_callback = 0;

void set_error_callback(ErrorCallback callback) {
  error_callback = callback;
}

void warn_if_not_overridden(const char* function,const char* file,unsigned int line,const type_info& type) {
  Log::cerr<<format("*** GEODE_WARNING: %s:%s:%d: Function not overridden by %s",file,function,line,type.name())<<endl;
}

void warning(const string& message,const char* function,const char* file,unsigned int line) {
  Log::cerr<<format("*** GEODE_WARNING: %s:%s:%d: %s",file,function,line,message)<<endl;
}

void function_is_not_defined(const char* function,const char* file,unsigned int line,const type_info& type) {
  const string error = format("%s:%s:%d: Function not defined by %s",file,function,line,type.name());
  if (error_callback) {
    error_callback(error);
#ifdef _WIN32
    throw RuntimeError("error callback returned: "+error);
#endif
  } else
    throw RuntimeError(error);
}

void not_implemented(const char* function,const char* file,unsigned int line,const char* message) {
  const string error = format("%s:%s:%d: Not implemented: %s",file,function,line,message?message:"something");
  if (error_callback) {
    error_callback(error);
#ifdef _WIN32
    throw RuntimeError("error callback returned: "+error);
#endif
  } else
    throw NotImplementedError(error);
}

void fatal_error(const char* function,const char* file,unsigned int line,const char* message) {
  const string error = format("%s:%s:%d: %s",file,function,line,message?message:"Fatal error");
  if (error_callback) {
    error_callback(error);
#ifdef _WIN32
    throw RuntimeError("error callback returned: "+error);
#endif
  } else
    throw RuntimeError(error);
}

void assertion_failed(const char* function,const char* file,unsigned int line,const char* condition,const char* message) {
  const string error = format("%s:%s:%d: %s, condition = %s",file,function,line,message?message:"Assertion failed",condition);
  static const bool break_on_assert = getenv("GEODE_BREAK_ON_ASSERT")!=0;
  if (break_on_assert) {
    Log::cout<<flush;
    Log::cerr<<"\n";
    process::backtrace();
    Log::cerr<<"\n*** Error: "<<error<<'\n'<<endl;
    BREAKPOINT();
  }
  if (error_callback) {
    error_callback(error);
#ifdef _WIN32
    throw RuntimeError("error callback returned: "+error);
#endif
  } else
    throw AssertionError(error);
}

void did_not_raise(const type_info& error, const char* message) {
  throw AssertionError(message?message:format("Expected %s, but function completed normally",error.name()));
}

}
