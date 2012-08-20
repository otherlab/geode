#pragma once

namespace other {

template<class Iter> struct Range {
  const Iter lo, hi;

  Range(const Iter& lo, const Iter& hi)
    :lo(lo),hi(hi) {}

  const Iter& begin() const { return lo; }
  const Iter& end() const { return hi; }
};

template<> struct Range<int> {
  int lo, hi;
  
  struct Iter {
    int i;
    explicit Iter(int i) : i(i) {}
    bool operator!=(Iter j) { return i != j.i; }
    void operator++() { i++; }
    int operator*() const { return i; }
  };

  Range()
    : lo(0), hi(0) {}

  Range(int lo, int hi)
    : lo(lo), hi(hi) {
    assert(lo<=hi);
  }

  int size() const { return hi-lo; }

  Iter begin() const { return Iter(lo); }
  Iter end() const { return Iter(hi); }
};

template<class Iter> static inline Range<Iter> range(const Iter& lo, const Iter& hi) {
  return Range<Iter>(lo,hi);
}

static inline Range<int> range(int n) {
  return Range<int>(0,n);
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
