//#####################################################################
// Class Random
//#####################################################################
#pragma once

#include <other/core/random/counter.h>
#include <other/core/array/view.h>
#include <other/core/python/Object.h>
#include <other/core/vector/Vector.h>
#include <boost/cstdint.hpp>
#include <boost/noncopyable.hpp>
#include <cmath>
#include <ctime>
namespace other {

using ::std::log;
using ::std::ldexp;
template<class TV> class Box;

class Random : public Object {
public:
  OTHER_DECLARE_TYPE
  typedef Object Base;
  typedef mpl::if_c<sizeof(real)==4,uint32_t,uint64_t>::type RealBits;
  BOOST_STATIC_ASSERT(sizeof(RealBits)==sizeof(real));

  uint128_t seed; // Counter mode, so we get to expose key as a mutable field!
private:
  uint128_t counter;
  int free_bit_count;
  uint128_t free_bits;
  real free_gaussian; // If nonzero, a free Gaussian random number

protected:
  explicit Random(uint128_t seed) OTHER_EXPORT;
public:
  ~Random();

  template<class Int> Int bits() OTHER_EXPORT;

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

  real normal() OTHER_EXPORT;

  real exponential(real mean) { // We take mean = 1/lambda for speed
    return -mean*log(uniform());
  }

  template<class TV> TV unit_sphere() {
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

  template<class TV> Rotation<TV> rotation() {
    return rotation_helper(unit_sphere<Vector<real,2*TV::m-2> >());
  }

  template<class TV> Frame<TV> frame(const TV& v0,const TV& v1) {
    TV v = uniform<TV>(v0,v1);
    return Frame<TV>(v,rotation<TV>());
  }

  template<class TV> Twist<TV> twist(const real& a) {
    Twist<TV> tw;
    tw.set_vector(uniform<Vector<real,Twist<TV>::m>>(-a,a));
    return tw;
  }

  template<class TArray> void shuffle(TArray& v) {
    int n = v.size();
    for (int i=0;i<n-1;i++) {
      int j = uniform<int>(i,n);
      if (i!=j) swap(v[i],v[j]);
    }
  }

private:
  template<class S> static Rotation<Vector<S,1> > rotation_helper(const Vector<S,0>&) {
    return Rotation<Vector<S,1> >();
  }

  template<class S> static Rotation<Vector<S,2> > rotation_helper(const Vector<S,2>& v) {
    return Rotation<Vector<S,2> >::from_complex(Complex<S>(v));
  }

  template<class S> static Rotation<Vector<S,3> > rotation_helper(const Vector<S,4>& v) {
    return Rotation<Vector<S,3> >::from_quaternion(Quaternion<S>(v));
  }
public:

  real uniform() { // in [0,1)
    return ldexp((real)1,-8*(int)sizeof(real))*bits<RealBits>();
  }

  Array<real> normal_py(int size);
  Array<real> uniform_py(int size);
  Array<int> uniform_int_py(int lo, int hi, int size);
};

// In [a,b)
template<>  int16_t Random::uniform(const  int16_t a, const  int16_t b);
template<> uint16_t Random::uniform(const uint16_t a, const uint16_t b);
template<>  int32_t Random::uniform(const  int32_t a, const  int32_t b);
template<> uint32_t Random::uniform(const uint32_t a, const uint32_t b);
template<>  int64_t Random::uniform(const  int64_t a, const  int64_t b);
template<> uint64_t Random::uniform(const uint64_t a, const uint64_t b);

template<> inline real Random::uniform(const real a, const real b) { // in [a,b)
  assert(a<b);
  return a+ldexp(b-a,-8*(int)sizeof(real))*bits<RealBits>();
}

}
