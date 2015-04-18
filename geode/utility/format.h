//#####################################################################
// Function format
//#####################################################################
//
// Similar to boost::format, with the following differences:
//
// 1. Not as powerful (no format("%g")%vector) or as safe.
// 2. More concise.
// 3. Doesn't add 100k to every object file.
//
// The main advantage over raw varargs is that we can pass string, and
// we can add more safety features later if we feel like it.
//
//#####################################################################
#pragma once

#include <geode/utility/config.h>
#include <geode/utility/type_traits.h>
#include <string>
#include <cinttypes>
namespace geode {

// format("Print things of type size_t as x = %" GEODE_PRIUSIZE " or similar", x)
// This macro selects printf formatting characters for size_t since they will depend on OS and architecture
// Warning: Not for use in PyString_FromFormat or similar which may expect different characters on Windows
#if GEODE_SIZEOF_SIZE_T == 4
  #define GEODE_PRIUSIZE PRIu32
#elif GEODE_SIZEOF_SIZE_T == 8
  #define GEODE_PRIUSIZE PRIu64
#else
  #error "Unable to set GEODE_PRIUSIZE"
#endif

using std::string;

// Unfortunately, since format_helper is called indirectly through format, we won't get much benefit from gcc's format attribute
GEODE_CORE_EXPORT string format_helper(const char* format,...) GEODE_FORMAT_PRINTF(1,2);

template<class T> static inline typename mpl::if_<is_enum<T>,int,T>::type format_sanitize(const T d) {
  static_assert(mpl::or_<is_fundamental<T>,is_enum<T>,is_pointer<T>>::value,"Passing as a vararg is not safe");
  return d;
}

static inline const char* format_sanitize(char* s) {
  return s;
}

static inline const char* format_sanitize(const char* s) {
  return s;
}

static inline const char* format_sanitize(const string& s) {
  return s.c_str();
}

#ifdef GEODE_VARIADIC

// Using this would allow gcc to check format string against other arguments...until you use an expression with a comma:
// #define format(fmt,...) format_helper(fmt, GEODE_MAP(format_sanitize, __VA_ARGS__))

// The format attribute sanity checks the format string, but can't verify it was used with a matching types in Args (only supports varargs)
template<class... Args> GEODE_FORMAT_PRINTF(1,0) static inline string format(const char* format, const Args&... args) {
  GEODE_GNUC_ONLY(_Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wformat-nonliteral\""))
  return format_helper(format,format_sanitize(args)...);
  GEODE_GNUC_ONLY(_Pragma("GCC diagnostic pop"))
}

#else // Unpleasant nonvariadic versions

static inline string format(const char* format) {
  return format_helper(format);
}

template<class A0> static inline string format(const char* format, const A0& a0) {
  return format_helper(format,format_sanitize(a0));
}

template<class A0,class A1> static inline string format(const char* format, const A0& a0, const A1& a1) {
  return format_helper(format,format_sanitize(a0),format_sanitize(a1));
}

template<class A0,class A1,class A2> static inline string format(const char* format, const A0& a0, const A1& a1, const A2& a2) {
  return format_helper(format,format_sanitize(a0),format_sanitize(a1),format_sanitize(a2));
}

template<class A0,class A1,class A2,class A3> static inline string format(const char* format, const A0& a0, const A1& a1, const A2& a2, const A3& a3) {
  return format_helper(format,format_sanitize(a0),format_sanitize(a1),format_sanitize(a2),format_sanitize(a3));
}

template<class A0,class A1,class A2,class A3,class A4> static inline string format(const char* format, const A0& a0, const A1& a1, const A2& a2, const A3& a3, const A4& a4) {
  return format_helper(format,format_sanitize(a0),format_sanitize(a1),format_sanitize(a2),format_sanitize(a3),format_sanitize(a4));
}

template<class A0,class A1,class A2,class A3,class A4,class A5> static inline string format(const char* format, const A0& a0, const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5) {
  return format_helper(format,format_sanitize(a0),format_sanitize(a1),format_sanitize(a2),format_sanitize(a3),format_sanitize(a4),format_sanitize(a5));
}

template<class A0,class A1,class A2,class A3,class A4,class A5,class A6> static inline string format(const char* format,
  const A0& a0, const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6) {
  return format_helper(format,format_sanitize(a0),format_sanitize(a1),format_sanitize(a2),format_sanitize(a3),format_sanitize(a4),format_sanitize(a5),format_sanitize(a6));
}

template<class A0,class A1,class A2,class A3,class A4,class A5,class A6,class A7> static inline string format(const char* format,
  const A0& a0, const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7) {
  return format_helper(format,format_sanitize(a0),format_sanitize(a1),format_sanitize(a2),format_sanitize(a3),format_sanitize(a4),format_sanitize(a5),format_sanitize(a6),format_sanitize(a7));
}

template<class A0,class A1,class A2,class A3,class A4,class A5,class A6,class A7,class A8> static inline string format(const char* format,
  const A0& a0, const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8) {
  return format_helper(format,format_sanitize(a0),format_sanitize(a1),format_sanitize(a2),format_sanitize(a3),format_sanitize(a4),format_sanitize(a5),format_sanitize(a6),format_sanitize(a7),format_sanitize(a8));
}

#endif

}
