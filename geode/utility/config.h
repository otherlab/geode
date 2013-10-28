//#####################################################################
// Header config
//#####################################################################
#pragma once

#include <geode/python/config.h> // Must be included first

namespace geode {
#if defined(GEODE_FLOAT)
typedef float real;
#elif defined(GEODE_DOUBLE)
typedef double real;
#else // default to double
#define GEODE_DOUBLE
typedef double real;
#endif
}

#define GEODE_NO_EXPORT // For documentation purposes

#ifndef _WIN32

#define GEODE_VARIADIC

// Mark the current symbol for export
#define GEODE_EXPORT __attribute__ ((visibility("default")))
// Mark the current symbol as imported (does nothing in non-windows)
#define GEODE_IMPORT

#define GEODE_HIDDEN __attribute__ ((visibility("hidden")))

#define GEODE_UNUSED __attribute__ ((unused))
#define GEODE_NORETURN(declaration) declaration __attribute__ ((noreturn))
#define GEODE_WARN_UNUSED_RESULT __attribute__ ((warn_unused_result))
#define GEODE_NEVER_INLINE __attribute__ ((noinline))
#define GEODE_PURE __attribute__ ((pure))
#define GEODE_CONST __attribute__ ((const))
#define GEODE_FORMAT(type,fmt,list) __attribute__ ((format(type,fmt,list)))
#define GEODE_EXPECT(value,expect) __builtin_expect((value),expect)

#if defined(__GNUC__) && __GNUC__ > 3 && defined(__GNUC_MINOR__) && __GNUC_MINOR__ > 4
#define GEODE_UNREACHABLE() ({ GEODE_DEBUG_ONLY(GEODE_FATAL_ERROR();) __builtin_unreachable(); })
#else
#define GEODE_UNREACHABLE() GEODE_FATAL_ERROR()
#endif

#ifdef NDEBUG
#  define GEODE_ALWAYS_INLINE __attribute__ ((always_inline))
#else
#  define GEODE_ALWAYS_INLINE
#endif

#if !defined(__clang__)
#  define GEODE_COLD __attribute__ ((cold))
#else
#  define GEODE_COLD
#endif

#if defined(NDEBUG) && !defined(__clang__)
#  define GEODE_FLATTEN __attribute__ ((flatten))
#else
#  define GEODE_FLATTEN
#endif

#ifdef __clang__
#  define GEODE_HAS_FEATURE(feature) __has_feature(feature)
#else
#  define GEODE_HAS_FEATURE(feature) false
#endif

#if !defined(__clang__) || GEODE_HAS_FEATURE(cxx_noexcept)
#  define GEODE_NOEXCEPT noexcept
#else
#  define GEODE_NOEXCEPT
#endif

#define GEODE_ALIGNED(n) __attribute__((aligned(n)))

#else // _WIN32

#ifndef GEODE_SINGLE_LIB
#define GEODE_EXPORT __declspec(dllexport)
#define GEODE_IMPORT __declspec(dllimport)
#else
// These should be defined to be empty, but Windows complains about empty macro arguments.
#define GEODE_EXPORT __declspec()
#define GEODE_IMPORT __declspec()
#endif

#define GEODE_HIDDEN
#define GEODE_UNUSED
#define GEODE_NORETURN(declaration) __declspec(noreturn) declaration
#define GEODE_WARN_UNUSED_RESULT
#define GEODE_ALWAYS_INLINE
#define GEODE_FLATTEN
#define GEODE_NEVER_INLINE
#define GEODE_PURE
#define GEODE_CONST
#define GEODE_UNREACHABLE() GEODE_FATAL_ERROR()
#define GEODE_NOEXCEPT
#define GEODE_COLD
#define GEODE_FORMAT
#define GEODE_EXPECT(value,expect) (value)
#define GEODE_ALIGNED(n) __declspec(align(n))

#endif

// Mark symbols when building geode
#ifdef _WIN32

#ifdef BUILDING_geode
#define GEODE_CORE_EXPORT GEODE_EXPORT
#else
#define GEODE_CORE_EXPORT GEODE_IMPORT
#endif

// no class exports on windows (only cause trouble, and also conflict with member exports)
#define GEODE_CORE_CLASS_EXPORT

#else

// On non-windows, this is not a import/export issue, but a visibility issue.
// So we don't have to distinguish between import and export.
#define GEODE_CORE_EXPORT GEODE_EXPORT

// On non-windows, typeid needs to be exported for each class
#define GEODE_CORE_CLASS_EXPORT GEODE_EXPORT

#endif
