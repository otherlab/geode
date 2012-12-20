// Locate extra resource files (textures, etc.) at runtime
#include <vector>

#include <other/core/utility/resource.h>
#include <other/core/array/Array.h>
#include <other/core/python/module.h>
#include <other/core/python/utility.h>
#if defined(_WIN32)
#define WINDOWS_MEAN_AND_LEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <unistd.h>
#endif
namespace other {

// Grab a path to the current executable
static string executable_path() 
{
#if defined(_WIN32)

#if defined(_UNICODE) || defined(UNICODE)
    typedef std::wstring str;
#else
    typedef std::string str;
#endif
  str path(MAX_PATH, 0);
  LPTSTR buffer = const_cast<LPTSTR>(path.data());
  size_t m = GetModuleFileName(0, buffer, MAX_PATH-1);
#if defined(_UNICODE) || defined(UNICODE)
#pragma message("Need to verify if non-ASCII characters are handled as UTF-8")
  const str::value_type* start = path.data();
  const std::string result(start, start + m + 1);
#else
  const str result(path);
#endif
  return result.data();

#elif defined(__APPLE__)
  char small[128];
  uint32_t size = sizeof(small)-1;
  if (!_NSGetExecutablePath(small,&size))
    return small;
  Array<char> large(size,false); 
  OTHER_ASSERT(!_NSGetExecutablePath(large.data(),&size)); 
  return large.data();
#elif defined(__linux__)
  for (size_t n=128;;n*=2) {
    Array<char> path(n,false);
    ssize_t m = readlink("/proc/self/exe",path.data(),n-1);
    if (m<0)
      throw OSError(format("executable_path: readlink failed, %s",strerror(errno)));
    if (m<n-1)
      return path.data();
  }
#else
  OTHER_NOT_IMPLEMENTED("Don't know how to extract path to exe on this platform");
#endif
}

static string& helper() {
  static string path = path::dirname(executable_path());
  return path;
}

string resource_path() {
  return helper();
}

string resource(const string& path) {
  return path::join(helper(),path);
}

string resource(const string& path0, const string& path1) {
  return path::join(helper(),path0,path1);
}

string resource(const string& path0, const string& path1, const string& path2) {
  return path::join(helper(),path0,path1,path2);
}

}
using namespace other;

void wrap_resource() {
#ifdef OTHER_PYTHON
  // Python is active, so set the executable path to the script directory
  Ref<> sys = steal_ref_check(PyImport_ImportModule("sys"));
  Ref<> argv = python_field(sys,"argv"); 
  Ref<> argv0 = steal_ref_check(PySequence_GetItem(&*argv,0)); 
  helper() = path::dirname(from_python<string>(argv0));
#endif
  OTHER_FUNCTION(resource_path)
  OTHER_FUNCTION_2(resource,static_cast<string(*)(const string&)>(resource))
}
