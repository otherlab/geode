//#####################################################################
// Class Random
//#####################################################################
#include <other/core/random/Random.h>
#include <other/core/python/Class.h>
#include <other/core/vector/Rotation.h>
#include <other/core/python/stl.h>
namespace other {

OTHER_DEFINE_TYPE(Random)
using std::vector;

Random::Random(uint128_t seed)
  : seed(seed)
  , counter(0)
  , free_bit_count(0)
  , free_bits(0)
  , free_gaussian(0) {}

Random::~Random() {}

template<class Int> Int Random::bits() {
  const int width = 8*sizeof(Int);
  if (free_bit_count<width) {
    free_bits = threefry(seed,counter++);
    free_bit_count = 128;
  }
  free_bit_count -= width;
  Int r = free_bits;
  free_bits >>= width;
  return r;
}

template uint16_t Random::bits();
template uint32_t Random::bits();
template uint64_t Random::bits();

#define INT(I,UI) \
  template<> I Random::uniform(const I a, const I b) { \
    assert(a<b); \
    const UI n = b-a; \
    /* Pick the largest possible multiple of n (minus 1) for rejection sampling */ \
    const UI limit = (UI(-n)/n)*n+n-1; \
    for (;;) { \
      const UI bits = this->bits<UI>(); \
      if (bits<=limit) \
        return a+bits%n; \
    } \
  }
INT( int16_t,uint16_t)
INT(uint16_t,uint16_t)
INT( int32_t,uint32_t)
INT(uint32_t,uint32_t)
INT( int64_t,uint64_t)
INT(uint64_t,uint64_t)
#undef INT

real Random::normal() {
  if (free_gaussian) {
    real r = free_gaussian;
    free_gaussian = 0;
    return r;
  }
  for (;;) {
    real v0 = uniform<real>(-1,1);
    real v1 = uniform<real>(-1,1);
    real s = sqr(v0)+sqr(v1);
    if (s && s<1) {
      real scale = sqrt(-2*log(s)/s);
      free_gaussian = scale*v1;
      return scale*v0;
    }
  }
}

// The result should consist of all (dependent) binomially distributed random values
static vector<Array<int>> random_bits_test(Random& random, int steps) {
  vector<Array<int>> all;
  #define WIDTH(w) \
    { \
      typedef uint##w##_t UI; \
      Array<int> raw(w); \
      Array<int> pairs(w*(w-1)/2); \
      for (int s=0;s<steps;s++) { \
        const UI n = random.bits<UI>(); \
        int k = 0; \
        for (int i=0;i<w;i++) { \
          raw[i] += (n>>i)&1; \
          for (int j=i+1;j<w;j++) \
            pairs[k++] += ((n>>i)&1)^((n>>j)&1); \
        } \
      } \
      all.push_back(raw); \
      all.push_back(pairs); \
      Array<Vector<int,9>> lags(w); \
      UI prev[10]; \
      for (int i=0;i<9;i++) \
        prev[i] = random.bits<UI>(); \
      for (int s=0;s<steps;s++) { \
        const UI n = random.bits<UI>(); \
        prev[9] = n; \
        for (int i=0;i<9;i++) { \
          const UI x = n^prev[i]; \
          prev[i] = prev[i+1]; \
          for (int j=0;j<w;j++) \
            lags[j][i] += (x>>i)&1; \
        } \
      } \
      all.push_back(scalar_view_own(lags)); \
    }
  WIDTH(16)
  WIDTH(32)
  WIDTH(64)
  return all;
}

Array<real> Random::normal_py(int size) {
  Array<real> result(size,false);
  for (auto& x : result)
    x = normal();
  return result;
}

Array<real> Random::uniform_py(int size) {
  Array<real> result(size,false);
  for (auto& x : result)
    x = uniform();
  return result;
}

Array<int> Random::uniform_int_py(int lo, int hi, int size) {
  Array<int> result(size,false);
  for (auto& x : result)
    x = uniform<int>(lo,hi);
  return result;
}

}
using namespace other;

void wrap_Random() {
  typedef Random Self;
  Class<Self>("Random")
    .OTHER_INIT(uint128_t)
    .OTHER_FIELD(seed)
    .OTHER_METHOD_2("normal",normal_py)
    .OTHER_METHOD_2("uniform",uniform_py)
    .OTHER_METHOD_2("uniform_int",uniform_int_py)
    ;

  OTHER_FUNCTION(random_bits_test)
}
