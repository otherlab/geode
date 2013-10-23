// Stream I/O related utilities

#pragma once

#include <geode/utility/format.h>
#include <istream>
namespace geode {

using std::istream;

struct expect {
  char c;
  expect(char c)
    :c(c) {}
};

GEODE_CORE_EXPORT void GEODE_NORETURN(throw_unexpected_error(expect expected,char got));

inline istream& operator>>(istream& input, expect expected) {
  char got;
  input >> got;
  if (got != expected.c)
    throw_unexpected_error(expected,got);
  return input;
}

}
