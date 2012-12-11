// Stream I/O related utilities

#pragma once

#include <other/core/utility/format.h>
#include <istream>
namespace other {

using std::istream;

struct expect {
  char c;
  expect(char c)
    :c(c) {}
};

OTHER_CORE_EXPORT void OTHER_NORETURN(throw_unexpected_error(expect expected,char got));

inline istream& operator>>(istream& input, expect expected) {
  char got;
  input >> got;
  if (got != expected.c)
    throw_unexpected_error(expected,got);
  return input;
}

}
