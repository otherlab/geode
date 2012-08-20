// Stream I/O related utilities 

#include <other/core/utility/stream.h>
#include <other/core/python/exceptions.h>
namespace other {

void throw_unexpected_error(expect expected,char got) {
  throw ValueError(format("expected '%c' during stream input, got '%c'",expected.c,got));
}

}
