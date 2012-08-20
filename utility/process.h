//#####################################################################
// Namespace process
//#####################################################################
#pragma once

#include <other/core/utility/config.h>
#ifndef Win32
#include <fenv.h>
#endif
namespace other {
namespace process {

size_t memory_usage() OTHER_EXPORT;

#ifdef Win32
enum {
  FE_INVALID   = 0x01,
  FE_DIVBYZERO = 0x04,
  FE_OVERFLOW  = 0x08,
  FE_UNDERFLOW = 0x10,
  FE_INEXACT   = 0x20
};
#endif

// exceptions should be a combination of FE_DIVBYZERO, FE_INEXACT, FE_INVALID, FE_OVERFLOW, FE_UNDERFLOW
void set_float_exceptions(const int exceptions=FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW) OTHER_EXPORT;

void backtrace() OTHER_EXPORT;
void set_backtrace(const bool enable=true) OTHER_EXPORT;
void block_interrupts(const bool block=true) OTHER_EXPORT;

}
}
