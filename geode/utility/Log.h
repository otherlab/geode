//#####################################################################
// Class Log
//#####################################################################
#pragma once

#include <geode/utility/config.h>
#include <geode/utility/format.h>
#include <boost/noncopyable.hpp>
#include <ostream>
#include <string>
namespace geode {

using std::string;
using std::ostream;

namespace Log {

GEODE_CORE_EXPORT ostream& cout_Helper();
GEODE_CORE_EXPORT ostream& cerr_Helper();

static ostream& cout GEODE_UNUSED=cout_Helper();
static ostream& cerr GEODE_UNUSED=cerr_Helper();

GEODE_CORE_EXPORT void push_scope(const string& name);
GEODE_CORE_EXPORT void pop_scope();
GEODE_CORE_EXPORT void copy_to_file(const string& filename,const bool append);

GEODE_CORE_EXPORT bool initialized();
GEODE_CORE_EXPORT void configure(const string& root_name, const bool suppress_cout=false, const bool suppress_timing=false,const int verbosity_level=1<<30);
GEODE_CORE_EXPORT void cache_initial_output();
GEODE_CORE_EXPORT void finish();
GEODE_CORE_EXPORT void stop_time();
template<class TValue> GEODE_CORE_EXPORT void stat(const string& label, const TValue& value);
GEODE_CORE_EXPORT void reset();
GEODE_CORE_EXPORT void dump();

namespace {
struct Scope : private boost::noncopyable {
public:
  Scope(const string& name) {
    push_scope(name);
  }

  ~Scope() {
    pop_scope();
  }
};
}

GEODE_CORE_EXPORT bool is_timing_suppressed();
GEODE_CORE_EXPORT void time_helper(const string& label);

#ifdef GEODE_VARIADIC

template<class... Args> static inline void time(const char* fmt, Args&&... args) {
  if (!is_timing_suppressed()) time_helper(format(fmt,args...));
}

#else // Unpleasant nonvariadic versions

static inline void time(const char* fmt) { if (!is_timing_suppressed()) time_helper(format(fmt)); }
template<class A0> static inline void time(const char* fmt, A0&& a0) { if (!is_timing_suppressed()) time_helper(format(fmt,a0)); }
template<class A0,class A1> static inline void time(const char* fmt, A0&& a0, A1&& a1) { if (!is_timing_suppressed()) time_helper(format(fmt,a0,a1)); }
template<class A0,class A1,class A2> static inline void time(const char* fmt, A0&& a0, A1&& a1, A2&& a2) { if (!is_timing_suppressed()) time_helper(format(fmt,a0,a1,a2)); }

#endif

}
}
