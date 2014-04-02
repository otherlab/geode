// Ranges of integers or iterators
#pragma once

#include <geode/utility/config.h>
#include <geode/utility/forward.h>
#include <geode/utility/type_traits.h>
#include <geode/utility/validity.h>
#include <cassert>
namespace geode {

GEODE_VALIDITY_CHECKER(has_subtract,T,declval<T>()-declval<T>())

template<class Iter,class Enable> struct Range {
  typedef decltype(*declval<Iter>()) reference;
  typedef typename mpl::if_<has_subtract<Iter>,Iter,int>::type sub;
  typedef decltype(declval<sub>()-declval<sub>()) Size;

  Iter lo, hi;

  Range() {}

  Range(const Iter& lo, const Iter& hi)
    : lo(lo), hi(hi) {}

  const Iter& begin() const { return lo; }
  const Iter& end() const { return hi; }

  Size size() const { return hi-lo; }
  template<class I> reference operator[](const I i) const { assert(i<hi-lo); return *(lo+i); }

  reference front() const { assert(lo!=hi); return *lo; }
  reference back() const { assert(lo!=hi); return *(hi-1); }
};

template<class I> struct Range<I,typename enable_if<is_integral<I>>::type> {
  I lo, hi;

  struct Iter {
    I i;
    explicit Iter(I i) : i(i) {}
    bool operator!=(Iter j) const { return i != j.i; }
    void operator++() { ++i; }
    void operator--() { --i; }
    I operator*() const { return i; }
  };

  Range()
    : lo(), hi() {}

  Range(const I lo, const I hi)
    : lo(lo), hi(hi) {
    assert(lo<=hi);
  }

  bool empty() const { return hi == lo; }
  I size() const { return hi-lo; }

  Iter begin() const { return Iter(lo); }
  Iter end() const { return Iter(hi); }

  I operator[](const I i) const { assert(0<=i && i<hi-lo); return lo+i; }

  I front() const { assert(lo<hi); return lo; }
  I back() const { assert(lo<hi); return hi-1; }

  bool contains(I i) const { return lo <= i && i < hi; }

  bool operator==(const Range r) const { return lo==r.lo && hi==r.hi; }
  bool operator!=(const Range r) const { return lo!=r.lo || hi!=r.hi; }
};

template<class Iter> static inline Range<Iter> range(const Iter& lo, const Iter& hi) {
  return Range<Iter>(lo,hi);
}

template<class I> static inline Range<I> range(I n) {
  static_assert(is_integral<I>::value,"Single argument range must take an integral type");
  return Range<I>(0,n);
}

template<class Iter,class I> static inline auto operator+(const Range<Iter>& r, const I n)
  -> typename enable_if<is_integral<I>,decltype(range(r.lo+n,r.hi+n))>::type {
  return range(r.lo+n,r.hi+n);
}

template<class Iter,class I> static inline auto operator-(const Range<Iter>& r, const I n)
  -> typename enable_if<is_integral<I>,decltype(range(r.lo-n,r.hi-n))>::type {
  return range(r.lo-n,r.hi-n);
}

template<class Iter,class I> static inline auto operator+(const I n, const Range<Iter>& r)
  -> typename enable_if<is_integral<I>,decltype(range(r.lo+n,r.hi+n))>::type {
  return range(r.lo+n,r.hi+n);
}

}
