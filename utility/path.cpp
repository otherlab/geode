// Path convenience functions

#include <other/core/utility/path.h>
#include <other/core/utility/format.h>
#include <other/core/python/exceptions.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <stdexcept>
#include <iostream>

namespace other {
namespace path {

string shell_quote(const string& s) {
#ifdef _WIN32
  std::string result = "\"";
  for(const char c : s) {
    if(c <= 0x1f || c >= 0x7f)
      throw RuntimeError(format("no support for escaping non-printable character 0x%X", (unsigned int)c));

    switch(c) {
      case '"':
        result += "\\\"";
        break;
      case '%':
        result += "%%";
        break;
      default:
        result += c;
    }
  }
  result += "\"";
  return result;
#else
  std::string result = "'";
  for(const char c : s) {
    if(c <= 0x1f || c >= 0x7f)
      throw RuntimeError(format("no support for escaping non-printable character 0x%X", (unsigned int)c));
    
    switch(c) {
      case '\'':
        result += "'\\''";
        break;
      default:
        result += c;
    }
  }
  result += '\'';
  return result;
#endif
}

string join(const string& p, const string& q) {
  return p+sep+q;
}

string join(const string& p0, const string& p1, const string& p2) {
  return join(join(p0,p1),p2);
}

string join(const string& p0, const string& p1, const string& p2, const string& p3) {
  return join(join(p0,p1),join(p2,p3));
}

// WARNING: These do not work for escaped directory separators
string extension(const string& path) {
  for (int i=path.size()-1;i>=0;i--) {
    // we stop if we're at the beginning, or at the first directory separator
    // this will treat filenames that begin with a '.' (and have no other '.')
    // as having no extension
    if (!i || is_sep(path[i-1]))
      break;
    if (path[i]=='.')
      return path.substr(i);
  }
  return string();
}

string remove_extension(const string& path) {
  string ext = extension(path);
  if (ext.empty())
    return path;
  return path.substr(0,path.size()-ext.size());
}

string basename(const string& path) {
  for (int i=path.size()-1;i>=0;i--)
    if (is_sep(path[i]))
      return path.substr(i+1);
  return path;
}

string dirname(const string& path) {
  for (int i=path.size()-1;i>=0;i--)
    if (is_sep(path[i]))
      return path.substr(0,i);
  return string();
}

void copy_file(const string &from, const string &to) {
  int ret = 0;
#ifdef _WIN32
  BOOL fail_if_exists = FALSE;
  if (!CopyFileA(from.c_str(), to.c_str(), fail_if_exists)) {
    ret = GetLastError();
  }
#else
  // WARNING: the file names are not escaped, so this is a severe security risk.
  string cmd = format("cp '%s' '%s'", from, to);
  //std::cout << "running shell command: " << cmd << std::endl;
  ret = system(cmd.c_str());
#endif

  if (ret)
    throw IOError(format("error %d while copying '%s' to '%s'.", ret, from, to));
}

}
}
