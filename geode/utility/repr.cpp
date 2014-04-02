//#####################################################################
// Function repr
//#####################################################################
#include <geode/utility/repr.h>
#include <geode/utility/format.h>
#include <cstring>
#include <cstdio>
namespace geode {

string repr(const float x) {
  static char buffer[40];
  sprintf(buffer,"%.9g",x);
  return buffer;
}

string repr(const double x) {
  static char buffer[40];
  sprintf(buffer,"%.17g",x);
  return buffer;
}

string repr(const long double x) {
  static char buffer[40];
  sprintf(buffer,"%.21Lg",x);
  return buffer;
}

static string str_repr(const char* s, const size_t n) {
  string r;
  r.push_back('\'');
  for (size_t i=0;i!=n;i++) {
    const char c = s[i];
    switch (c) {
      case '\t': r.push_back('\\'); r.push_back('t'); break;
      case '\n': r.push_back('\\'); r.push_back('n'); break;
      case '\r': r.push_back('\\'); r.push_back('r'); break;
      case '\'': r.push_back('\\'); r.push_back('\''); break;
      case '\\': r.push_back('\\'); r.push_back('\\'); break;
      default:
        if (' '<=c && c<='~')
          r.push_back(c);
        else {
          r.push_back('\\');
          r.push_back('x');
          const uint8_t a = uint8_t(c)/16, b = c&15;
          r.push_back(a+(a<10?'0':'a'-10));
          r.push_back(b+(b<10?'0':'a'-10));
        }
    }
  }
  r.push_back('\'');
  return r;
}

string repr(const string& s) {
  return str_repr(s.c_str(),s.size());
}

string repr(const char* s) {
  return str_repr(s,strlen(s));
}

// For testing purposes
string str_repr_test(const string& s) {
  return repr(s);
}

}
