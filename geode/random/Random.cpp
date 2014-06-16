//#####################################################################
// Class Random
//#####################################################################
#include <geode/random/Random.h>
#include <geode/python/Class.h>
#include <geode/vector/Frame.h>
#include <geode/vector/Rotation.h>
#include <geode/python/stl.h>
namespace geode {

GEODE_DEFINE_TYPE(Random)
using std::vector;

Random::Random(uint128_t seed)
  : seed(seed)
  , counter(0)
  , free_bit_count(0)
  , free_bits(0)
  , free_gaussian(0) {}

Random::~Random() {}

template<class Int, int N> Int Random::n_bits() {
  const int width = N;
  if (free_bit_count<width) {
    free_bits = threefry(seed,counter++);
    free_bit_count = 128;
  }
  free_bit_count -= width;
  Int r = cast_uint128<Int>(free_bits);
  free_bits >>= width;
  return r;
}

template<class Int> Int Random::bits() { return n_bits<Int, 8*sizeof(Int)>(); }
template<> bool Random::bits<bool>() { return n_bits<uint8_t, 1>() & 1; }

bool Random::bit() { return bits<bool>(); }

template  uint8_t Random::bits();
template uint16_t Random::bits();
template uint32_t Random::bits();
template unsigned long Random::bits();
template unsigned long long Random::bits();

#define INT(I,UI) \
  template<> I Random::uniform(const I a, const I b) { \
    assert(a<b); \
    const UI n = b-a; \
    /* Pick the largest possible multiple of n (minus 1) for rejection sampling */ \
    const UI limit = (UI(1+~n)/n)*n+n-1; \
    for (;;) { \
      const UI bits = this->bits<UI>(); \
      if (bits<=limit) \
        return a+bits%n; \
    } \
  }
INT(  int8_t, uint8_t)
INT( uint8_t, uint8_t)
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
  Array<real> result(size,uninit);
  for (auto& x : result)
    x = normal();
  return result;
}

Array<real> Random::uniform_py(int size) {
  Array<real> result(size,uninit);
  for (auto& x : result)
    x = uniform();
  return result;
}

Array<int> Random::uniform_int_py(int lo, int hi, int size) {
  Array<int> result(size,uninit);
  for (auto& x : result)
    x = uniform<int>(lo,hi);
  return result;
}

static Rotation<Vector<real,2> > rotation_helper(const Vector<real,2>& v) {
  return Rotation<Vector<real,2>>::from_complex(v.complex());
}

static Rotation<Vector<real,3> > rotation_helper(const Vector<real,4>& v) {
  return Rotation<Vector<real,3>>::from_quaternion(Quaternion<real>(v));
}

template<class TV> Rotation<TV> Random::rotation() {
  return rotation_helper(unit_ball<Vector<real,2*TV::m-2> >());
}

template<class TV> Frame<TV> Random::frame(const TV& v0,const TV& v1) {
  TV v = uniform(v0,v1);
  return Frame<TV>(v,rotation<TV>());
}

#define INSTANTIATE(d) \
  template GEODE_CORE_EXPORT Rotation<Vector<real,d>> Random::rotation(); \
  template GEODE_CORE_EXPORT Frame<Vector<real,d>> Random::frame(const Vector<real,d>&,const Vector<real,d>&);
INSTANTIATE(2)
INSTANTIATE(3)

}
using namespace geode;

void wrap_Random() {
  typedef Random Self;
  Class<Self>("Random")
    .GEODE_INIT(uint128_t)
    .GEODE_FIELD(seed)
    .GEODE_METHOD_2("normal",normal_py)
    .GEODE_METHOD_2("uniform",uniform_py)
    .GEODE_METHOD_2("uniform_int",uniform_int_py)
    ;

  GEODE_FUNCTION(random_bits_test)
}
