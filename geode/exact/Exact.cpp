// Multiprecision integer arithmetic for exact geometric predicates

#include <geode/exact/Exact.h>
#include <geode/array/alloca.h>
#include <geode/array/Subarray.h>
#include <geode/python/wrap.h>
#include <geode/random/Random.h>
namespace geode {

using std::cout;
using std::endl;

RawArray<mp_limb_t> trim(RawArray<mp_limb_t> x) {
  int n = x.size();
  for (;n>0;n--)
    if (x[n-1])
      break;
  return x.slice(0,n);
}

RawArray<const mp_limb_t> trim(RawArray<const mp_limb_t> x) {
  int n = x.size();
  for (;n>0;n--)
    if (x[n-1])
      break;
  return x.slice(0,n);
}


string mpz_str(RawArray<const mp_limb_t> limbs, const bool hex) {
  GEODE_ASSERT(limbs.size());
  if (mp_limb_signed_t(limbs.back())<0) { // Negative
    const auto neg = GEODE_RAW_ALLOCA(limbs.size(),mp_limb_t);
    mpn_neg(neg.data(),limbs.data(),limbs.size());
    return "-"+mpz_str(neg,hex);
  } else { // Nonnegative
    int n = limbs.size();
    for (;n>0;n--)
      if (limbs[n-1])
        break;
    if (!n) // Zero
      return "0";
    else { // Positive
      const auto s = GEODE_RAW_ALLOCA(hex ? 5+2*sizeof(mp_limb_t)*n
                                          : 3+int(ceil(8*sizeof(mp_limb_t)*log10(2)*n)),unsigned char);
      auto p = s.data()+2*hex;
      const int count = (int) mpn_get_str(p,hex?16:10,limbs.slice(0,n).copy().data(),n);
      for (int i=0;i<count;i++)
        p[i] += p[i]<10?'0':'a'-10;
      p[count] = 0;
      if (hex) {
        s[0] = '0';
        s[1] = 'x';
      }
      return (const char*)s.data();
    }
  }
}

string mpz_str(Subarray<const mp_limb_t,2> limbs, const bool hex) {
  string s;
  s += '[';
  for (int i=0;i<limbs.m;i++) {
    if (i)
      s += ',';
    s += mpz_str(limbs[i],hex);
  }
  s += ']';
  return s;
}

template<int a> static string hex(const Exact<a> x) {
  return mpz_str(asarray(x.n),true);
}

template<int a> static inline Exact<a> random_exact(Random& random) {
  Exact<a> x(uninit);
  for (int i=0;i<x.limbs;i++)
    x.n[i] = random.bits<mp_limb_t>();
  return x;
}

static void fast_exact_tests() {
#if !GEODE_FAST_EXACT
  std::cout << "No fast exact arithmetic" << std::endl;
#else
  const auto random = new_<Random>(1823131);
  for (int i=0;i<32;i++) {
    #define SAME(a) { \
      const auto x = random_exact<a>(random), \
                 y = random_exact<a>(random), \
                 p = x + y, \
                 m = x - y, \
                 s = x << 2; \
      const auto xx = sqr(x); \
      Exact<a> r(uninit); \
      mpn_add_n(r.n,x.n,y.n,r.limbs); \
      GEODE_ASSERT(r == p,format("add %d:\n    x %s\n    y %s\n  x+y %s\n    r %s",a,hex(x),hex(y),hex(p),hex(r))); \
      mpn_sub_n(r.n,x.n,y.n,r.limbs); \
      GEODE_ASSERT(r == m,format("sub %d:\n    x %s\n    y %s\n  x-y %s\n    r %s",a,hex(x),hex(y),hex(m),hex(r))); \
      mpn_lshift(r.n,x.n,x.limbs,2); \
      GEODE_ASSERT(r == s); \
      Exact<2*a> rr(uninit); \
      const auto ax = is_negative(x) ? -x : x; \
      mpn_sqr(rr.n,ax.n,x.limbs); \
      GEODE_ASSERT(rr == xx,format("sqr %d:\n   x %s\n  xx %s\n   r %s",a,hex(x),hex(xx),hex(rr))); }
    #define MUL(a,b) { \
      const auto x = random_exact<a>(random); \
      const auto y = random_exact<b>(random); \
      const auto xy = x*y; \
      Exact<a+b> r(uninit); \
      mpn_mul(r.n,x.n,x.limbs,y.n,y.limbs); \
      if (is_negative(x)) mpn_sub_n(r.n+x.limbs,r.n+x.limbs,y.n,y.limbs); \
      if (is_negative(y)) mpn_sub_n(r.n+y.limbs,r.n+y.limbs,x.n,x.limbs); \
      GEODE_ASSERT(r == xy,format("mul %d %d:\n   x %s\n   y %s\n  xy %s\n   r %s", \
                                  a,b,hex(x),hex(y),hex(xy),hex(r))); }
    SAME(1)
    SAME(2)
    SAME(3)
    MUL(1,1)
    MUL(2,2)
    MUL(3,3)
    MUL(2,1)
    MUL(3,1)
    MUL(3,2)
  }
#endif
}

#if 0 // Single operation routines for easy inspection of assembly

Exact<2> inspect_two_limb_add(const Exact<2> x, const Exact<2> y) { return x+y; }
Exact<4> inspect_four_limb_add(const Exact<4> x, const Exact<4> y) { return x+y; }
Exact<4> inspect_four_limb_sub(const Exact<4> x, const Exact<4> y) { return x-y; }
Exact<8> inspect_four_limb_mul(const Exact<4> x, const Exact<4> y) { return x*y; }

#endif

}
using namespace geode;

void wrap_exact_exact() {
  GEODE_FUNCTION(fast_exact_tests)
}
