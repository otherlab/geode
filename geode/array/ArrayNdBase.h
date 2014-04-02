//#####################################################################
// Class ArrayNdBase
//#####################################################################
#pragma once

#include <geode/array/Array.h>
#include <geode/math/isnan.h>
#include <geode/utility/type_traits.h>
namespace geode {

template<class T,class TArray>
class ArrayNdBase {
public:
  typedef T Element;
  typedef typename ScalarPolicy<T>::type Scalar;
  static const bool is_const = geode::is_const<T>::value;

  Array<T> flat; // one-dimensional data storage

  template<class... Args> ArrayNdBase(const Args&... args)
    : flat(args...) {}

  TArray& derived() {
    return static_cast<TArray&>(*this);
  }

  const TArray& derived() const {
    return static_cast<const TArray&>(*this);
  }

  const shared_ptr<const Owner>& owner() const {
    return flat.owner();
  }

  T* data() const {
    return flat.data();
  }

  bool operator==(const TArray& v) const {
    return derived().sizes()==v.sizes() && flat==v.flat;
  }

  bool operator!=(const TArray& v) const {
    return !(*this==v);
  }

  TArray& operator+=(const TArray& v) {
    assert(derived().sizes()==v.sizes());
    flat += v.flat;
    return derived();
  }

  TArray& operator+=(const T& a) {
    flat += a;
    return derived();
  }

  TArray& operator-=(const TArray& v) {
    assert(derived().sizes()==v.sizes());
    flat -= v.flat;
    return derived();
  }

  TArray& operator-=(const T& a) {
    flat -= a;
    return derived();
  }

  template<class T2,class TArrayT2> TArray& operator*=(const ArrayNdBase<T2,TArrayT2>& v) {
    assert(derived().sizes()==v.derived().sizes());
    flat *= v.flat;
    return derived();
  }

  template<class T2> typename enable_if<mpl::or_<IsScalar<T2>,is_same<T,T2> >,TArray&>::type operator*=(const T2 a) {
    flat *= a;
    return derived();
  }

  template<class T2> TArray& operator/=(const T2 a) {
    return *this *= inverse(a);
  }

  int count_true() const {
    return flat.count_true();
  }

  void fill(const T& constant) const {
    flat.fill(constant);
  }

  static void copy(const TArray& old_copy, TArray& new_copy) {
    assert(old_copy.sizes()==new_copy.sizes());
    Array<T>::copy(old_copy.flat,new_copy.flat);
  }

  template<class T2> static void copy(const T2 constant,const TArray& old_copy, TArray& new_copy) {
    assert(old_copy.sizes()==new_copy.sizes());
    new_copy.flat = constant*old_copy.flat;
  }

  template<class T2> static void copy(const T2 c1, const TArray& v1, const TArray& v2, TArray& result) {
    assert(v1.sizes()==v2.sizes() && v2.sizes()==result.sizes());
    result.flat = c1*v1.flat+v2.flat;
  }

  template<class T2> static void copy(const T2 c1,const TArray& v1,const T2 c2,const TArray& v2,TArray& result) {
    assert(v1.sizes()==v2.sizes() && v2.sizes()==result.sizes());
    result.flat = c1*v1.flat+c2*v2.flat;
  }

  template<class T2> static void copy(const T2 c1,const TArray& v1,const T2 c2,const TArray& v2,const T2 c3,const TArray& v3,TArray& result) {
    assert(v1.sizes()==v2.sizes() && v2.sizes()==v3.sizes() && v3.sizes()==result.sizes());
    result.flat = c1*v1.flat+c2*v2.flat+c3*v3.flat;
  }

  void clamp_below(const T& value) {
    flat.clamp_below(value);
  }

  T average() const {
    return flat.average();
  }

  T max() const {
    return flat.max();
  }

  T maxabs() const {
    return flat.maxabs();
  }

  T maxmag() const {
    return flat.maxmag();
  }

  T min() const {
    return flat.min();
  }

  T minmag() const {
    return flat.minmag();
  }

  T sum() const {
    return flat.sum();
  }

  T sumabs() const {
    return flat.sumabs();
  }

  T componentwise_maxabs() const {
    return flat.componentwise_maxabs();
  }

  static T dot(const TArray& a1, const TArray& a2) {
    assert(a1.sizes()==a2.sizes());
    return Array<T>::dot(a1.flat,a2.flat);
  }

  Scalar max_magnitude() {
    return flat.max_magnitude();
  }
};

}
