//#####################################################################
// Namespace process
//#####################################################################
#include <other/core/utility/process.h>
#include <other/core/utility/debug.h>
#include <other/core/utility/Log.h>
#include <other/core/math/min.h>
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <cstring>
#ifndef _WIN32
#include <sys/time.h>
#include <sys/resource.h>
#include <signal.h>
#include <execinfo.h>
#endif
#if defined(__APPLE__) && defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace other{
namespace process{

using std::endl;
using std::flush;

#ifdef _WIN32

size_t memory_usage(){OTHER_NOT_IMPLEMENTED();}
void set_float_exceptions(const int exceptions){OTHER_NOT_IMPLEMENTED();}
void backtrace(){OTHER_NOT_IMPLEMENTED();}
void set_backtrace(const bool enable){OTHER_NOT_IMPLEMENTED();}
void block_interrupts(const bool block){OTHER_NOT_IMPLEMENTED();}

#else

#ifdef __linux__

// Recent versions of Linux no longer implement getrusage usefully, so read from /proc/self/statm instead
size_t memory_usage() {
  FILE* file = fopen("/proc/self/statm","r");
  if (!file)
    return 0;
  size_t size = 0;
  int r = fscanf(file,"%zu",&size);
  fclose(file);
  return r==1?getpagesize()*size:0;
}

#else

// On other versions of posix, we have a convenient portable system call.
size_t memory_usage() {
  struct rusage usage;
  getrusage(RUSAGE_SELF,&usage);
  return usage.ru_idrss+usage.ru_isrss;
}

#endif

static void float_exception_handler(int sig_number, siginfo_t* info, void *data) {
  if (sig_number!=SIGFPE) OTHER_FATAL_ERROR();
  Log::cerr<<"** Error: SIGNAL "<<"SIGFPE ("<<sig_number<<") **"<<endl;
  Log::cerr<<"Floating point exception: reason "<<info->si_code<<" = \""<<
      (info->si_code==FPE_INTDIV?"integer divide by zero":info->si_code==FPE_INTOVF?"integer overflow":
      info->si_code==FPE_FLTDIV?"FP divide by zero":info->si_code==FPE_FLTOVF?"FP overflow":
      info->si_code==FPE_FLTUND?"FP underflow":info->si_code==FPE_FLTRES?"FP inexact result":
      info->si_code==FPE_FLTINV?"FP invalid operation":info->si_code==FPE_FLTSUB?"subscript out of range":"unknown")
      << "\", from address 0x"<<std::hex<<(unsigned long)info->si_addr<<endl;
  backtrace();
  Log::finish();
  exit(sig_number);
}

#ifdef __APPLE__
#ifdef __SSE__

// feenableexcept and fedisableexcept are not defined, so define them

static int flags_to_mask(int flags) {
  int result=0;
  if (flags&FE_INVALID) result|=_MM_MASK_INVALID;
  if (flags&FE_DIVBYZERO) result|=_MM_MASK_DIV_ZERO;
  if (flags&FE_OVERFLOW) result|=_MM_MASK_OVERFLOW;
  if (flags&FE_UNDERFLOW) result|=_MM_MASK_UNDERFLOW;
  if (flags&FE_INEXACT) result|=_MM_MASK_INEXACT;
  return result;
}

static void fedisableexcept(int flags) {
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() | flags_to_mask(flags));
}

static void feenableexcept(int flags) {
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~flags_to_mask(flags));
}

#else // No SSE

static void fedisableexcept(int flags) { OTHER_NOT_IMPLEMENTED(); }
static void feenableexcept(int flags) { OTHER_NOT_IMPLEMENTED(); }

#endif
#endif

void set_float_exceptions(const int exceptions) {
  static bool have_original_action = false;
  static struct sigaction original_action;
  if (!have_original_action) // initialize with original action
    sigaction(SIGFPE,0,&original_action);
  if (exceptions) {
    // Avoid catching delayed exceptions caused by external code
    fedisableexcept(FE_ALL_EXCEPT);
    feclearexcept(exceptions);
    // install new handler
    struct sigaction action;
    action.sa_flags=SA_SIGINFO;
    action.sa_sigaction=float_exception_handler;
    sigemptyset(&action.sa_mask);
    if(sigaction(SIGFPE,&action,0)) OTHER_FATAL_ERROR("Could not register Fpe signal handler");
    feenableexcept(exceptions);
  } else {
    if (sigaction(SIGFPE,&original_action,0)) OTHER_FATAL_ERROR("Could not restore Fpe signal handler");
    fedisableexcept(FE_ALL_EXCEPT);
  }
}

void backtrace() {
  const int stack_entries = 50;
  void* stack_array[stack_entries];
  const int size = (int)::backtrace(stack_array,stack_entries);
  char** strings = backtrace_symbols(stack_array,size);
  Log::cerr << "=================== Begin Stack Backtrace ===================" << endl;
  // Note: We used to use c++filt here, but the popen call would occasionally hang.  It's possible this was
  // a bug in the code, but having a dying process hang instead of quit is extremely bad, so I'm removing
  // the c++filt for now.
  for (int i=0;i<size;i++) {
    // Apple's backtrace prints out symbols beginning with _ZN, but c++filt wants __ZN
#ifdef __APPLE__
    char* z = strstr(strings[i],"_ZN");
#else
    char* z = 0;
#endif
    if (z) {
      *z = 0;
      Log::cerr << strings[i] << "__" << z+1 << '\n';
    } else
      Log::cerr << strings[i] << '\n';
  }
  Log::cerr << "==================== End Stack Backtrace ====================" << endl;
  free(strings);
}

static bool caught_interrupt_signal = false;

static void interrupt_signal_handler(int signal_id) {
  caught_interrupt_signal = true;
}

void block_interrupts(const bool block) {
  static bool have_original_action = false;
  static struct sigaction original_action;
  if (block) {
    if (have_original_action) OTHER_FATAL_ERROR("Nested call to block_interrupts(true).");
    struct sigaction action;
    action.sa_flags = 0;
    action.sa_handler = interrupt_signal_handler;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT,&action,&original_action)) OTHER_FATAL_ERROR("Could not block interrupt signal.");
    have_original_action=true;
  } else {
    if (!have_original_action) OTHER_FATAL_ERROR("Call to block_interrupts(false) before block_interrupts(true).");
    if (sigaction(SIGINT,&original_action,0)) OTHER_FATAL_ERROR("Could not unblock interrupt signal.");
    if (caught_interrupt_signal) {
      Log::cerr<<"Caught delayed interrupt signal."<<endl;
      raise(SIGINT);
    }
    have_original_action=false;
  }
}

static int catch_signals[] = {SIGINT,SIGABRT,SIGSEGV,SIGBUS,SIGTERM,SIGHUP,SIGUSR2,0};
static const char *catch_signal_names[] = {"SIGINT","SIGABRT","SIGSEGV","SIGBUS","SIGTERM","SIGHUP","SIGUSR2",(char*)0};

void backtrace_and_abort(int signal_number, siginfo_t* info, void *data) {
  Log::cout<<flush;
  Log::cerr<<"\n";
  backtrace();
  const char** names = catch_signal_names;
  const char *signal_name=0;
  for(int *i=catch_signals;*i!=0;i++,names++)
    if (signal_number==*i)
      signal_name = *names;
  Log::cerr<<"\n*** Error: SIGNAL "<<(signal_name?signal_name:"Unknown")<<" ("<<signal_number<<")\n"<<endl;
  if (signal_number!=SIGUSR2) {
    Log::finish();
    exit(signal_number);
  }
}

void set_backtrace(const bool enable) {
  if (enable) {
    struct sigaction action;
    action.sa_flags = SA_SIGINFO;
    action.sa_sigaction = backtrace_and_abort;
    sigemptyset(&action.sa_mask);
    for (int *i=catch_signals;*i!=0;i++)
      sigaddset(&action.sa_mask,*i);
    for (int *i=catch_signals;*i!=0;i++)
      if(sigaction(*i,&action,0))
        OTHER_FATAL_ERROR("Failed to install backtrace handler.");
  } else
    for (int *i=catch_signals;*i!=0;i++)
      signal(*i,SIG_DFL);
}

#endif
}
}
