// Multiprecision integer arithmetic for exact geometric predicates

#include <other/core/exact/Exact.h>
#include <other/core/array/alloca.h>
#include <other/core/array/Subarray.h>
namespace other {

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
  OTHER_ASSERT(limbs.size());
  if (mp_limb_signed_t(limbs.back())<0) { // Negative
    const auto neg = OTHER_RAW_ALLOCA(limbs.size(),mp_limb_t);
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
      const auto s = OTHER_RAW_ALLOCA(hex ? 5+2*sizeof(mp_limb_t)*n
                                          : 3+int(ceil(8*sizeof(mp_limb_t)*log10(2)*n)),unsigned char);
      auto p = s.data()+2*hex;
      const int count = mpn_get_str(p,hex?16:10,limbs.slice(0,n).copy().data(),n);
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

}
