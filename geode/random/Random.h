//#####################################################################
// Class Random
//#####################################################################
#pragma once

#include <geode/random/counter.h>
#include <geode/array/view.h>
#include <geode/utility/Object.h>
#include <geode/vector/Vector.h>
#include <cmath>
#include <ctime>
namespace geode {

using ::std::log;
using ::std::ldexp;
template<class TV> class Box;

class Random : public Object {
public:
  GEODE_NEW_FRIEND
  typedef Object Base;
  typedef mpl::if_c<sizeof(real)==4,uint32_t,uint64_t>::type RealBits;
  static_assert(sizeof(RealBits)==sizeof(real),"");

  uint128_t seed; // Counter mode, so we get to expose key as a mutable field!
private:
  uint128_t counter;
  int free_bit_count;
  uint128_t free_bits;
  real free_gaussian; // If nonzero, a free Gaussian random number

protected:
  GEODE_CORE_EXPORT explicit Random(uint128_t seed);
public:
  ~Random();

  template<class Int> GEODE_CORE_EXPORT Int bits();

  GEODE_CORE_EXPORT bool bit();

  template<class S> S uniform(const typename ScalarPolicy<S>::type a,const typename ScalarPolicy<S>::type b) { // in [a,b)
    // Specialized versions for int and real are below
    S r;fill_uniform(r,a,b);return r;
  }

  template<class TV> void fill_uniform(TV& v,const typename ScalarPolicy<TV>::type a,const typename ScalarPolicy<TV>::type b) {
    typedef typename ScalarPolicy<TV>::type Scalar;
    RawArray<Scalar> sv = scalar_view(v);
    for (int i=0;i<sv.size();i++) sv[i] = uniform<Scalar>(a,b);
  }

  template<class S,int d> Vector<S,d> uniform(const Vector<S,d>& min,const Vector<S,d>& max) {
    Vector<S,d> r;
    for (int i=0;i<d;i++) r[i] = uniform<S>(min[i],max[i]);
    return r;
  }

  template<class T,int d> Vector<T,d> uniform(const Box<Vector<T,d> >& box) {
    return uniform(box.min,box.max);
  }

  GEODE_CORE_EXPORT real normal();

  real exponential(real mean) { // We take mean = 1/lambda for speed
    return -mean*log(uniform());
  }

  template<class TV> TV unit_ball() {
    for (;;) {
      TV v = uniform<TV>(-1,1);
      if (v.sqr_magnitude()<=1) return v;
    }
  }

  template<class TV> TV direction() {
    if (!TV::m) return TV();
    for (;;) {
      TV v = uniform<TV>(-1,1);
      typename TV::Scalar sqr_magnitude=v.sqr_magnitude();
      if (sqr_magnitude>0 && sqr_magnitude<=1) return v/sqrt(sqr_magnitude);
    }
  }

  template<class TArray> void shuffle(TArray& v) {
    int n = v.size();
    for (int i=0;i<n-1;i++) {
      int j = uniform<int>(i,n);
      if (i!=j) swap(v[i],v[j]);
    }
  }

  real uniform() { // in [0,1)
    return ldexp((real)1,-8*(int)sizeof(real))*bits<RealBits>();
  }

  Array<real> normal_py(int size);
  Array<real> uniform_py(int size);
  Array<int> uniform_int_py(int lo, int hi, int size);
  template<class TV> GEODE_CORE_EXPORT Rotation<TV> rotation();
  template<class TV> GEODE_CORE_EXPORT Frame<TV> frame(const TV& v0,const TV& v1);
private:
  template<class Int, int N> Int n_bits();
};

// In [a,b)
template<> GEODE_CORE_EXPORT   int8_t Random::uniform(const   int8_t a, const   int8_t b);
template<> GEODE_CORE_EXPORT  uint8_t Random::uniform(const  uint8_t a, const  uint8_t b);
template<> GEODE_CORE_EXPORT  int16_t Random::uniform(const  int16_t a, const  int16_t b);
template<> GEODE_CORE_EXPORT uint16_t Random::uniform(const uint16_t a, const uint16_t b);
template<> GEODE_CORE_EXPORT  int32_t Random::uniform(const  int32_t a, const  int32_t b);
template<> GEODE_CORE_EXPORT uint32_t Random::uniform(const uint32_t a, const uint32_t b);
template<> GEODE_CORE_EXPORT  int64_t Random::uniform(const  int64_t a, const  int64_t b);
template<> GEODE_CORE_EXPORT uint64_t Random::uniform(const uint64_t a, const uint64_t b);

template<> inline real Random::uniform(const real a, const real b) { // in [a,b)
  assert(a<b);
  return a+ldexp(b-a,-8*(int)sizeof(real))*bits<RealBits>();
}

// For testing purposes
GEODE_CORE_EXPORT vector<Array<int>> random_bits_test(Random& random, const int steps);

}
