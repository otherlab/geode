//#####################################################################
// Macro OTHER_DEBUG_PRINT
//#####################################################################
#pragma once

#include <other/core/utility/config.h>
#include <other/core/utility/overload.h>
#include <other/core/utility/str.h>
#include <other/core/utility/stl.h>
#include <other/core/utility/macro_map.h>
namespace other {

void debug_print_single(const char* prefix) OTHER_CORE_EXPORT;
void debug_print_msg(const char* prefix, const char* msg) OTHER_CORE_EXPORT;
void debug_print_helper(const char* prefix,...) OTHER_CORE_EXPORT;
void debug_print_helper_multiline(const char* prefix,...) OTHER_CORE_EXPORT;

#define OTHER_DEBUG_PRINT_HELPER(a) #a,other::str(a).c_str()

#define OTHER_DEBUG_PRINT(prefix,...) other::debug_print_helper(prefix,OTHER_MAP(OTHER_DEBUG_PRINT_HELPER,__VA_ARGS__),(char*)0)
#define OTHER_DEBUG_PRINT_MULTILINE(prefix,...) other::debug_print_helper_multiline(prefix,OTHER_MAP(OTHER_DEBUG_PRINT_HELPER,__VA_ARGS__),(char*)0)

#define OTHER_STRINGIZE_HELPER(s) #s
#define OTHER_STRINGIZE(s) OTHER_STRINGIZE_HELPER(s)

#define OTHER_HERE_STRING() ((std::string(__FILE__ ":" OTHER_STRINGIZE(__LINE__) ":")+std::string(__FUNCTION__)).c_str())
#define OTHER_TRACE()  other::debug_print_single(OTHER_HERE_STRING())
#define OTHER_TRACE_MSG(msg)  other::debug_print_msg(OTHER_HERE_STRING(),(": " + other::str(msg)).c_str())
#define OTHER_PROBE(...)  OTHER_DEBUG_PRINT(OTHER_HERE_STRING(),__VA_ARGS__)
#define OTHER_PROBE_MULTILINE(...)  OTHER_DEBUG_PRINT_MULTILINE(OTHER_HERE_STRING(),__VA_ARGS__)

#define OTHER_ASSERT_PROBE(condition,...) if(condition){}else{ OTHER_PROBE(__VA_ARGS__); OTHER_FATAL_ERROR("Assertion Failed: "#condition);}
#define OTHER_ASSERT_PROBE_MULTILINE(condition,...) if(condition){}else{OTHER_PROBE_MULTILINE(__VA_ARGS__); OTHER_FATAL_ERROR("Assertion Failed: "#condition);}

#define OTHER_TRACE_IF(condition) if(condition){OTHER_TRACE();}else{}
#define OTHER_PROBE_IF(condition,...) if(condition){OTHER_PROBE(__VA_ARGS__);}else{}
#define OTHER_PROBE_IF_MULTILINE(condition,...) if(condition){OTHER_PROBE(__VA_ARGS__);}else{}

}
