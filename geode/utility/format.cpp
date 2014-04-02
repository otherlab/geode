//#####################################################################
// Function format
//#####################################################################
#include <geode/utility/format.h>
#include <vector>
#include <cstdarg>
#include <cstdio>

//These needed so test functions will compile without python
#include <geode/utility/debug.h>
#include <iostream>

// Windows silliness
#undef small
#undef far
#undef near

namespace geode {

string format_helper(const char* format,...) {
  // Try using a small buffer first
  va_list marker;
  va_start(marker,format);
  char small[64];
  int non_null_char_count = vsnprintf(small,sizeof(small)-1,format,marker);
  va_end(marker);
  if (unsigned(non_null_char_count) < sizeof(small)-1)
    return small;

#ifdef _WIN32
  // On Windows (and I think glibc pre 2.1), vsnprintf returns a useless negative number on failure,
  // we need to call a separate function to get the correct length.
  va_start(marker,format);
  non_null_char_count = _vscprintf(format,marker);
  va_end(marker);
#endif

  // Retry using (almost) exact buffer size
  //
  // vsnprintf specs differ by platform
  //   For osx/unix: The generated string has a length of at most n-1, leaving space for the additional terminating null character.
  //   For windows: If there is room at the end (that is, if the number of characters to write is less than count), the buffer will be null-terminated.
  //
  //  Both platforms will behave the same if the buffer has space for the terminating null character.
  //
  // It would be nice to store the data directly into a std::string and avoid a copy, but we use a vector in case we happen to use a std::string
  //   without contiguous storage (which isn't part of spec for C++98) or if there are any oddities from null characters in the middle of the string.

  std::vector<char> large(non_null_char_count + 1);
  va_start(marker,format);
  vsnprintf(&large[0],non_null_char_count + 1,format,marker);
  va_end(marker);

  return string(&large[0], non_null_char_count);
}

static void check_identity_format(const string& t) {
  const string result = format("%s", t);

  if(result != t) {
    std::cerr << "format(\"%s\", \"" << t << "\") returns " << result << std::endl;
    GEODE_FATAL_ERROR("format(\"%s\",t) != t");
  }

  // make sure nothing like extra null characters added on the end
  if(result.size() != t.size()) {
    std::cerr << "format(\"%s\", \"" << t << "\") returns " << result << std::endl;
    std::cerr << "test string size: " << t.size() << std::endl;
    std::cerr << "result size: " << result.size() << std::endl;
    GEODE_FATAL_ERROR("format(\"%s\",t).size() != t.size()");
  }
}

void format_test() {
  const string short_test_string = "this is a test";
  const string digits_62 = "12345678901234567890123456789012345678901234567890123456789012";
  const string digits_63 = "123456789012345678901234567890123456789012345678901234567890123";
  const string digits_64 = "1234567890123456789012345678901234567890123456789012345678901234";
  const string digits_65 = "12345678901234567890123456789012345678901234567890123456789012345";
  const string digits_66 = "123456789012345678901234567890123456789012345678901234567890123456";
  const string digits_70 = "1234567890123456789012345678901234567890123456789012345678901234567890";

  for(size_t size = 0; size < 100; ++size) {
    string test = format("%*s",size,"");
    GEODE_ASSERT(test.size() == size);
  }

  check_identity_format(short_test_string);
  check_identity_format(digits_62);
  check_identity_format(digits_63);
  check_identity_format(digits_64);
  check_identity_format(digits_65);
  check_identity_format(digits_66);
  check_identity_format(digits_62);
  check_identity_format(digits_70);
}

}
