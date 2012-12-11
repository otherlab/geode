//#####################################################################
// Header config
//#####################################################################
#pragma once

#include <other/core/python/config.h> // Must be included first

namespace other {
#if defined(OTHER_FLOAT)
typedef float real;
#elif defined(OTHER_DOUBLE)
typedef double real;
#else // default to double
#define OTHER_DOUBLE
typedef double real;
#endif
}

#ifndef _WIN32

#define OTHER_VARIADIC

// marks the current symbol for export
#define OTHER_EXPORT __attribute__ ((visibility("default")))
// marks the current symbol as imported (does nothing in non-windows)
#define OTHER_IMPORT

#define OTHER_HIDDEN __attribute__ ((visibility("hidden")))

#define OTHER_UNUSED __attribute__ ((unused))
#define OTHER_NORETURN(declaration) declaration __attribute__ ((noreturn))
#define OTHER_NEVER_INLINE __attribute__ ((noinline))
#define OTHER_PURE __attribute__ ((pure))
#define OTHER_CONST __attribute__ ((const))
#define OTHER_COLD __attribute__ ((const))
#define OTHER_FORMAT(type,fmt,list) __attribute__ ((format(type,fmt,list)))
#define OTHER_EXPECT(value,expect) __builtin_expect((value),expect)

#if defined(__GNUC__) && __GNUC__ > 3 && defined(__GNUC_MINOR__) && __GNUC_MINOR__ > 4
#define OTHER_UNREACHABLE() __builtin_unreachable()
#else
#define OTHER_UNREACHABLE() OTHER_FATAL_ERROR()
#endif

#ifdef NDEBUG
#  define OTHER_ALWAYS_INLINE __attribute__ ((always_inline))
#else
#  define OTHER_ALWAYS_INLINE
#endif

#if defined(NDEBUG) && !defined(__clang__)
#  define OTHER_FLATTEN __attribute__ ((flatten))
#else
#  define OTHER_FLATTEN
#endif

#ifdef __clang__
#  define OTHER_HAS_FEATURE(feature) __has_feature(feature)
#else
#  define OTHER_HAS_FEATURE(feature) false
#endif

#if !defined(__clang__) || OTHER_HAS_FEATURE(cxx_noexcept)
#  define OTHER_NOEXCEPT noexcept
#else
#  define OTHER_NOEXCEPT
#endif

#else // _WIN32

#define OTHER_EXPORT __declspec(dllexport)
#define OTHER_IMPORT __declspec(dllimport)

#define OTHER_HIDDEN
#define OTHER_UNUSED
#define OTHER_NORETURN(declaration) __declspec(noreturn) declaration
#define OTHER_ALWAYS_INLINE
#define OTHER_FLATTEN
#define OTHER_NEVER_INLINE
#define OTHER_PURE
#define OTHER_CONST
#define OTHER_UNREACHABLE() OTHER_FATAL_ERROR()
#define OTHER_NOEXCEPT
#define OTHER_COLD
#define OTHER_FORMAT
#define OTHER_EXPECT(value,expect) (value)

#endif

// Mark symbols when building core
#ifdef _WIN32

#ifdef BUILDING_other_core
#define OTHER_CORE_EXPORT OTHER_EXPORT
#else
#define OTHER_CORE_EXPORT OTHER_IMPORT
#endif

// no class exports on windows (only cause trouble, and also conflict with member exports)
#define OTHER_CORE_CLASS_EXPORT

#else

// On non-windows, this is not a import/export issue, but a visibility issue.
// So we don't have to distinguish between import and export.
#define OTHER_CORE_EXPORT OTHER_EXPORT

// On non-windows, typeid needs to be exported for each class
#define OTHER_CORE_CLASS_EXPORT OTHER_EXPORT

#endif
