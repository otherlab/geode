#pragma once

#include <geode/utility/config.h>
#include <geode/utility/forward.h>
#include <boost/utility/declval.hpp>
#include <cassert>
namespace geode {

template<class Iter,class Enable> struct Range {
  typedef decltype(*boost::declval<Iter>()) Reference;

  const Iter lo, hi;

  Range(const Iter& lo, const Iter& hi)
    :lo(lo),hi(hi) {}

  const Iter& begin() const { return lo; }
  const Iter& end() const { return hi; }

  Iter operator[](const int i) const { assert(0<=i && i<hi-lo); return lo+i; }

  Reference front() const { assert(lo!=hi); return *lo; }
  Reference back() const { assert(lo!=hi); return *(hi-1); }
};

template<class I> struct Range<I,typename boost::enable_if<boost::is_integral<I>>::type> {
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
    : lo(0), hi(0) {}

  Range(I lo, I hi)
    : lo(lo), hi(hi) {
    assert(lo<=hi);
  }

  I size() const { return hi-lo; }

  Iter begin() const { return Iter(lo); }
  Iter end() const { return Iter(hi); }

  I operator[](const I i) const { assert(0<=i && i<hi-lo); return lo+i; }

  I front() const { assert(lo<hi); return lo; }
  I back() const { assert(lo<hi); return hi-1; }

  bool contains(I i) const { return lo <= i && i < hi; }
};

template<class Iter> static inline Range<Iter> range(const Iter& lo, const Iter& hi) {
  return Range<Iter>(lo,hi);
}

template<class I> static inline Range<I> range(I n) {
  static_assert(boost::is_integral<I>::value,"single argument range must take an integral type");
  return Range<I>(0,n);
}

template<class Iter> static inline Range<Iter> operator+(const Range<Iter>& r, int n) {
  return Range<Iter>(r.lo+n,r.hi+n);
}

template<class Iter> static inline Range<Iter> operator-(const Range<Iter>& r, int n) {
  return Range<Iter>(r.lo-n,r.hi-n);
}

template<class Iter> static inline Range<Iter> operator+(int n, const Range<Iter>& r) {
  return Range<Iter>(r.lo+n,r.hi+n);
}

}
