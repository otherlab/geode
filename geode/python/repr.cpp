//#####################################################################
// Function repr
//#####################################################################
#include <geode/python/repr.h>
#include <geode/python/from_python.h>
#include <geode/utility/format.h>
#include <cstdio>
namespace geode {

string repr(PyObject& x) {
#ifdef GEODE_PYTHON
  return from_python<string>(steal_ref_check(PyObject_Repr(&x)));
#else
  return format("<object of type %s>",x.ob_type->tp_name);
#endif
}

string repr(PyObject* x) {
  return x ? repr(*x) : "None";
}

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

string repr(const string& s) {
  return repr(s.c_str());
}

string repr(const char* s) {
  string r;
  r.push_back('\'');
  while (const char c = *s++)
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
  r.push_back('\'');
  return r;
}

}
