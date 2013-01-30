//#####################################################################
// Class Log
//#####################################################################
#pragma once

#include <other/core/utility/config.h>
#include <other/core/utility/format.h>
#include <boost/noncopyable.hpp>
#include <ostream>
#include <string>
namespace other {

using std::string;
using std::ostream;

namespace Log {

OTHER_CORE_EXPORT ostream& cout_Helper();
OTHER_CORE_EXPORT ostream& cerr_Helper();

static ostream& cout OTHER_UNUSED=cout_Helper();
static ostream& cerr OTHER_UNUSED=cerr_Helper();

OTHER_CORE_EXPORT void push_scope(const string& name);
OTHER_CORE_EXPORT void pop_scope();
OTHER_CORE_EXPORT void copy_to_file(const string& filename,const bool append);

OTHER_CORE_EXPORT bool initialized();
OTHER_CORE_EXPORT void configure(const string& root_name, const bool suppress_cout=false, const bool suppress_timing=false,const int verbosity_level=1<<30);
OTHER_CORE_EXPORT void cache_initial_output();
OTHER_CORE_EXPORT void finish();
OTHER_CORE_EXPORT void stop_time();
template<class TValue> OTHER_CORE_EXPORT void stat(const string& label, const TValue& value);
OTHER_CORE_EXPORT void reset();
OTHER_CORE_EXPORT void dump();

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

OTHER_CORE_EXPORT bool is_timing_suppressed();
OTHER_CORE_EXPORT void time_helper(const string& label);

#ifdef OTHER_VARIADIC

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
