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

ostream& cout_Helper() OTHER_EXPORT;
ostream& cerr_Helper() OTHER_EXPORT;

static ostream& cout OTHER_UNUSED=cout_Helper();
static ostream& cerr OTHER_UNUSED=cerr_Helper();

void push_scope(const string& name) OTHER_EXPORT;
void pop_scope() OTHER_EXPORT;
void copy_to_file(const string& filename,const bool append) OTHER_EXPORT;

void configure(const string& root_name, const bool suppress_cout=false, const bool suppress_timing=false,const int verbosity_level=1<<30) OTHER_EXPORT;
void cache_initial_output() OTHER_EXPORT;
void finish() OTHER_EXPORT;
void stop_time() OTHER_EXPORT;
template<class TValue> OTHER_EXPORT void stat(const string& label, const TValue& value) OTHER_EXPORT;
void reset() OTHER_EXPORT;
void dump() OTHER_EXPORT;

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

bool is_timing_suppressed() OTHER_EXPORT;
void time_helper(const string& label) OTHER_EXPORT;

template<class... Args> static inline void time(const char* fmt, Args&&... args) {
  if(!is_timing_suppressed()) time_helper(format(fmt,args...));
}

}
}
