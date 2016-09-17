// Base 64 encoding and decoding

#include <geode/utility/base64.h>
#include <geode/python/wrap.h>
namespace geode {

using std::numeric_limits;

static inline char encode(uint8_t x) {
  return x<26 ? x+'A'
       : x<52 ? x-26+'a'
       : x<62 ? x-52+'0'
       : x==62 ? '+' : '/';
}

static inline uint32_t decode(char x) {
  if ('A'<=x && x<='Z') 
    return x-'A';
  if ('a'<=x && x<='z')
    return x-'a'+26;
  if ('0'<=x && x<='9')
    return x-'0'+52;
  if (x=='+')
    return 62;
  if (x=='/')
    return 63;
  throw ValueError(format("invalid character code %d in base 64 encoded string",x));
}

string base64_encode(const string& src) {
  GEODE_ASSERT(src.size() <= size_t(numeric_limits<int>::max()));
  return base64_encode(RawArray<const uint8_t>((int)src.size(),(const uint8_t*)src.c_str()));
}

string base64_encode(RawArray<const uint8_t> src) {
  string dst;
  dst.resize((src.size()+2)/3*4);
  const int n = src.size()/3;
  for (int i=0;i<n;i++) {
    const uint32_t x = uint32_t(src[3*i])<<16|uint32_t(src[3*i+1])<<8|src[3*i+2];
    dst[4*i+0] = encode(x>>18);
    dst[4*i+1] = encode(x>>12&63);
    dst[4*i+2] = encode(x>>6&63);
    dst[4*i+3] = encode(x&63);
  }
  // Handle remaining bytes
  if (src.size()==3*n+2) {
    const uint32_t x = uint32_t(src[3*n])<<16|uint32_t(src[3*n+1])<<8;
    dst[4*n+0] = encode(x>>18);
    dst[4*n+1] = encode(x>>12&63);
    dst[4*n+2] = encode(x>>6&63);
    dst[4*n+3] = '=';
  } else if (src.size()==3*n+1) {
    const uint32_t x = uint32_t(src[3*n])<<16;
    dst[4*n+0] = encode(x>>18);
    dst[4*n+1] = encode(x>>12&63);
    dst[4*n+2] = '=';
    dst[4*n+3] = '=';
  }
  return dst;
}

string base64_decode(const string& src) {
  string dst;
  if (!src.size())
    return dst;
  if (src.size()&3)
    throw ValueError(format("base 64 encoded string has length %d not a multiple of 4",src.size()));
  const auto dst_size = src.size()/4*3-(src[src.size()-1]=='=')-(src[src.size()-2]=='=');
  GEODE_ASSERT(dst_size<size_t(numeric_limits<int>::max()));
  dst.resize(int(dst_size));
  const int n = int((src.size()>>2)-(src[src.size()-1]=='='));
  for (int i=0;i<n;i++) {
    const uint32_t x = decode(src[4*i+0])<<18
                     | decode(src[4*i+1])<<12
                     | decode(src[4*i+2])<<6
                     | decode(src[4*i+3]);
    dst[3*i+0] = x>>16;
    dst[3*i+1] = x>>8&0xff;
    dst[3*i+2] = x&0xff;
  }
  // Handle remaining bytes
  if (dst.size()==size_t(3*n+2)) {
    const uint32_t x = decode(src[4*n+0])<<18
                     | decode(src[4*n+1])<<12
                     | decode(src[4*n+2])<<6;
    if (x&0xff)
      throw ValueError("bad ending to base 64 encoded string");
    dst[3*n+0] = x>>16;
    dst[3*n+1] = x>>8&0xff;
  } else if (dst.size()==size_t(3*n+1)) {
    const uint32_t x = decode(src[4*n+0])<<18
                     | decode(src[4*n+1])<<12;
    if (x&0xffff)
      throw ValueError("bad ending to base 64 encoded string");
    dst[3*n+0] = x>>16;
  }
  return dst;
}

}
using namespace geode;

void wrap_base64() {
  GEODE_FUNCTION_2(base64_encode,static_cast<string(*)(const string&)>(base64_encode))
  GEODE_FUNCTION(base64_decode)
}
