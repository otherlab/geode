//#####################################################################
// Namespace process
//#####################################################################
#pragma once

#include <other/core/utility/config.h>
#include <other/core/vector/forward.h>
#ifndef _WIN32
#include <fenv.h>
#endif
namespace other {
namespace process {

// CPU usage for (user,system)
OTHER_CORE_EXPORT Vector<double,2> cpu_times();

OTHER_CORE_EXPORT size_t memory_usage();
OTHER_CORE_EXPORT size_t max_memory_usage();

#ifdef _WIN32
enum {
  FE_INVALID   = 0x01,
  FE_DIVBYZERO = 0x04,
  FE_OVERFLOW  = 0x08,
  FE_UNDERFLOW = 0x10,
  FE_INEXACT   = 0x20
};
#endif

// Exceptions should be a combination of FE_DIVBYZERO, FE_INEXACT, FE_INVALID, FE_OVERFLOW, FE_UNDERFLOW
OTHER_CORE_EXPORT void set_float_exceptions(const int exceptions=FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW);

OTHER_CORE_EXPORT void backtrace();
OTHER_CORE_EXPORT void set_backtrace(const bool enable=true);
OTHER_CORE_EXPORT void block_interrupts(const bool block=true);

}
}
