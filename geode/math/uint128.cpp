//#####################################################################
// Class uint128_t
//#####################################################################
#include <geode/math/uint128.h>
#include <geode/utility/debug.h>
#include <geode/utility/format.h>
#include <iostream>
namespace geode {

using std::cout;

string str(uint128_t n) {
  const auto lo = cast_uint128<uint64_t>(n),
             hi = cast_uint128<uint64_t>(n>>64);
  // For now, we lazily produce hexadecimal to avoid having to divide.
  return hi ? format("0x%llx%016llx",hi,lo) : format("0x%llx",lo);
}

ostream& operator<<(ostream& output, uint128_t n) {
  return output << str(n);
}

#if defined(__GNUC__) && defined(__LP64__)
string str(__int128_t n) {
  return n<0 ? '-'+str(uint128_t(-n)) : str(uint128_t(n));
};
ostream& operator<<(ostream& output, __int128_t n) {
  return output << str(n);
}
#endif

vector<uint128_t> uint128_test(uint128_t x, uint128_t y) {
  // Test shifts
  static const int shifts[] = {0,1,63,64,65,127};
  for (const int s : shifts) {
    const auto p = uint128_t(1)<<s;
    GEODE_ASSERT((x<<s)==x*p);
    GEODE_ASSERT(((x>>s)<<s)==(x&~(p-1)));
  }
  // Test other operations
  std::vector<uint128_t> r;
  r.push_back((uint64_t)-7);
  r.push_back(-7);
  r.push_back(x);
  r.push_back(x+y);
  r.push_back(x-y);
  r.push_back(x*y);
  r.push_back(x<<5);
  r.push_back(x>>7);
  return r;
}

}
