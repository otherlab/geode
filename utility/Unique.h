#pragma once

#include <other/core/utility/move.h>
#include <other/core/utility/safe_bool.h>
#include <algorithm>
namespace other {

// Doesn't support custom deleter

template<typename T> class Unique {
public:
  typedef T element_type;
private:
  T* self;
public:
  Unique()
    : self(0) {}

  ~Unique() {
    delete self;
  }

  explicit Unique(T* ptr)
    : self(ptr) {}

  Unique(Unique&& rhs)
    : self(rhs.release()) {}

  template<typename S> Unique(Unique<S>&& rhs)
    : self(rhs.release()) {}

  Unique& operator=(Unique&& rhs) {
    delete self;
    self = rhs.release();
    return *this;
  }

  template<typename S> Unique& operator=(Unique<S>&& rhs) {
    delete self;
    self = rhs.release();
    return *this;
  }

  T* get() const {
    return self;
  }

  T& operator*() const {
    assert(self);
    return *self;
  }

  T* operator->() const {
    assert(self);
    return self;
  }

  void swap(Unique& rhs) {
    std::swap(self, rhs.self);
  }

  T* release() {
    T* p = self;
    self = 0;
    return p;
  }

  void reset() {
    delete self;
    self = 0;
  }

  void reset(T* new_ptr) {
    if (self != new_ptr) {
      delete self;
      self = new_ptr;
    }
  }

  operator SafeBool() const {
    return safe_bool(self);
  }

  Unique(const Unique& rhs) = delete;
  Unique& operator=(const Unique& rhs) = delete;
};

template<typename T> static inline void swap(Unique<T>& lhs, Unique<T>& rhs) { lhs.swap(rhs); }
template<typename T> static inline bool operator==(const Unique<T> &lhs, const Unique<T> &rhs) { return lhs.get() == rhs.get(); }
template<typename T> static inline bool operator!=(const Unique<T> &lhs, const Unique<T> &rhs) { return lhs.get() != rhs.get(); }
template<typename T> static inline bool operator<(const Unique<T> &lhs, const Unique<T> &rhs) { return lhs.get() < rhs.get(); }
template<typename T> static inline bool operator<=(const Unique<T> &lhs, const Unique<T> &rhs) { return lhs.get() <= rhs.get(); }
template<typename T> static inline bool operator>(const Unique<T> &lhs, const Unique<T> &rhs) { return lhs.get() > rhs.get(); }
template<typename T> static inline bool operator>=(const Unique<T> &lhs, const Unique<T> &rhs) { return lhs.get() >= rhs.get(); }

} // namespace other
