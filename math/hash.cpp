//#####################################################################
// Function hash
//#####################################################################
#include <other/core/math/hash.h>
#include <other/core/array/RawArray.h>
namespace other {

static Hash hash_string(size_t size,const char* s) {
  // TODO: process data in 64 bit blocks
  return hash_array(RawArray<const char>((int)size,s));
}

Hash hash_reduce(const string& s) {
  return hash_string(s.size(),s.c_str());
}

Hash hash_reduce(const char* s) {
  return hash_string(strlen(s),s);
}

}
