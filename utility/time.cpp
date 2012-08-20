// Current time

#include <other/core/utility/time.h>
#if defined(_WIN32)
#pragma comment(lib, "winmm.lib")
#include <windows.h>
#elif defined(__linux__) || defined(__CYGWIN__) || defined(__APPLE__)
#include <sys/time.h>
#include <sys/resource.h>
#endif
namespace other {

// Windows
#ifdef _WIN32

static double get_resolution() {
  __int64 frequency;
  QueryPerformanceFrequency((LargeInteger*)&frequency);
  return 1./frequency;
}
static double resolution = get_resolution();

double get_time() {
  __int64 time;
    QueryPerformanceCounter((LargeInteger*)&time);
    return resolution*time;
}

// Unix
#elif defined(__linux__) || defined(__CYGWIN__) || defined(__APPLE__)

double get_time() {
  timeval tv;
  gettimeofday(&tv,0);
  return tv.tv_sec+1e-6*tv.tv_usec;
}

#else

#error "Don't know how to get time on this platform"

#endif

}
