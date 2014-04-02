//#####################################################################
// Class Log
//#####################################################################
#pragma once

#include <geode/utility/config.h>
#include <geode/utility/format.h>
#include <geode/utility/forward.h>
#include <ostream>
#include <string>
namespace geode {

using std::string;
using std::ostream;

GEODE_CORE_EXPORT void log_push_scope(const string& name);
GEODE_CORE_EXPORT void log_pop_scope();
GEODE_CORE_EXPORT void log_copy_to_file(const string& filename,const bool append);

GEODE_CORE_EXPORT bool log_initialized();
GEODE_CORE_EXPORT void log_configure(const string& name, const bool suppress_cout=false,
                                     const bool suppress_timing=false, const int verbosity=10000);
GEODE_CORE_EXPORT void log_cache_initial_output();
GEODE_CORE_EXPORT void log_finish();
GEODE_CORE_EXPORT void log_stop_time();

template<class TValue> GEODE_CORE_EXPORT void log_stat(const string& label, const TValue& value);
GEODE_CORE_EXPORT void log_reset();
GEODE_CORE_EXPORT void log_dump();

GEODE_CORE_EXPORT bool log_is_timing_suppressed();
GEODE_CORE_EXPORT void log_time_helper(const string& label);

// For python (C++ should use Log::cout and Log::cerr)
GEODE_CORE_EXPORT void log_print(const string& str);
GEODE_CORE_EXPORT void log_error(const string& str);
GEODE_CORE_EXPORT void log_flush();

namespace Log {

GEODE_CORE_EXPORT ostream& cout_helper();
GEODE_CORE_EXPORT ostream& cerr_helper();
static ostream& cout GEODE_UNUSED=cout_helper();
static ostream& cerr GEODE_UNUSED=cerr_helper();

namespace {
struct Scope : private Noncopyable {
public:
  Scope(const string& name) {
    log_push_scope(name);
  }

  ~Scope() {
    log_pop_scope();
  }
};
}

#ifdef GEODE_VARIADIC

template<class... Args> static inline void time(const char* fmt, Args&&... args) {
  if (!log_is_timing_suppressed())
    log_time_helper(format(fmt,args...));
}

#else // Unpleasant nonvariadic versions

static inline void time(const char* fmt) { if (!log_is_timing_suppressed()) log_time_helper(format(fmt)); }
template<class A0> static inline void time(const char* fmt, A0&& a0) { if (!log_is_timing_suppressed()) log_time_helper(format(fmt,a0)); }
template<class A0,class A1> static inline void time(const char* fmt, A0&& a0, A1&& a1) { if (!log_is_timing_suppressed()) log_time_helper(format(fmt,a0,a1)); }
template<class A0,class A1,class A2> static inline void time(const char* fmt, A0&& a0, A1&& a1, A2&& a2) { if (!log_is_timing_suppressed()) log_time_helper(format(fmt,a0,a1,a2)); }

#endif

}
}
