// Stream I/O related utilities 

#include <geode/utility/stream.h>
#include <geode/utility/exceptions.h>
namespace geode {

void throw_unexpected_error(expect expected,char got) {
  throw ValueError(format("expected '%c' during stream input, got '%c'",expected.c,got));
}

}
