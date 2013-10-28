//#####################################################################
// Macro GEODE_DEBUG_PRINT
//#####################################################################
#pragma once

#include <geode/utility/config.h>
#include <geode/utility/overload.h>
#include <geode/utility/str.h>
#include <geode/utility/stl.h>
#include <geode/utility/macro_map.h>
namespace geode {

GEODE_CORE_EXPORT void debug_print_single(const char* prefix);
GEODE_CORE_EXPORT void debug_print_msg(const char* prefix, const char* msg);
GEODE_CORE_EXPORT void debug_print_helper(const char* prefix,...);
GEODE_CORE_EXPORT void debug_print_helper_multiline(const char* prefix,...);

#define GEODE_DEBUG_PRINT_HELPER(a) #a,geode::str(a).c_str()

#define GEODE_DEBUG_PRINT(prefix,...) geode::debug_print_helper(prefix,GEODE_MAP(GEODE_DEBUG_PRINT_HELPER,__VA_ARGS__),(char*)0)
#define GEODE_DEBUG_PRINT_MULTILINE(prefix,...) geode::debug_print_helper_multiline(prefix,GEODE_MAP(GEODE_DEBUG_PRINT_HELPER,__VA_ARGS__),(char*)0)

#define GEODE_STRINGIZE_HELPER(s) #s
#define GEODE_STRINGIZE(s) GEODE_STRINGIZE_HELPER(s)

#define GEODE_HERE_STRING() ((std::string(__FILE__ ":" GEODE_STRINGIZE(__LINE__) ":")+std::string(__FUNCTION__)).c_str())
#define GEODE_TRACE()  geode::debug_print_single(GEODE_HERE_STRING())
#define GEODE_TRACE_MSG(msg)  geode::debug_print_msg(GEODE_HERE_STRING(),(": " + geode::str(msg)).c_str())
#define GEODE_PROBE(...)  GEODE_DEBUG_PRINT(GEODE_HERE_STRING(),__VA_ARGS__)
#define GEODE_PROBE_MULTILINE(...)  GEODE_DEBUG_PRINT_MULTILINE(GEODE_HERE_STRING(),__VA_ARGS__)

#define GEODE_ASSERT_PROBE(condition,...) if(condition){}else{ GEODE_PROBE(__VA_ARGS__); GEODE_FATAL_ERROR("Assertion Failed: "#condition);}
#define GEODE_ASSERT_PROBE_MULTILINE(condition,...) if(condition){}else{GEODE_PROBE_MULTILINE(__VA_ARGS__); GEODE_FATAL_ERROR("Assertion Failed: "#condition);}

#define GEODE_TRACE_IF(condition) if(condition){GEODE_TRACE();}else{}
#define GEODE_PROBE_IF(condition,...) if(condition){GEODE_PROBE(__VA_ARGS__);}else{}
#define GEODE_PROBE_IF_MULTILINE(condition,...) if(condition){GEODE_PROBE(__VA_ARGS__);}else{}

#define DEBUG_TODO(...)
}
