#pragma once

#include <other/core/utility/move.h>
#include <other/core/utility/safe_bool.h>
#include <algorithm>
namespace other {

// Doesn't support custom deleter

template<typename T> class unique {
public:
  typedef T element_type;
private:
  T* self;
public:
  unique()
    : self(0) {}

  ~unique() {
    delete self;
  }

  explicit unique(T* ptr)
    : self(ptr) {}

  unique(unique&& rhs)
    : self(rhs.release()) {}

  template<typename S> unique(unique<S>&& rhs)
    : self(rhs.release()) {}

  unique& operator=(unique&& rhs) {
    delete self;
    self = rhs.release();
    return *this;
  }

  template<typename S> unique& operator=(unique<S>&& rhs) {
    delete self;
    self = rhs.release();
    return *this;
  }

  T* get() const {
    return self;
  }

  void swap(unique& rhs) {
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

  unique(const unique& rhs) = delete;
  unique& operator=(const unique& rhs) = delete;
};

template<typename T> static inline void swap(unique<T>& lhs, unique<T>& rhs) { lhs.swap(rhs); }
template<typename T> static inline bool operator==(const unique<T> &lhs, const unique<T> &rhs) { return lhs.get() == rhs.get(); }
template<typename T> static inline bool operator!=(const unique<T> &lhs, const unique<T> &rhs) { return lhs.get() != rhs.get(); }
template<typename T> static inline bool operator<(const unique<T> &lhs, const unique<T> &rhs) { return lhs.get() < rhs.get(); }
template<typename T> static inline bool operator<=(const unique<T> &lhs, const unique<T> &rhs) { return lhs.get() <= rhs.get(); }
template<typename T> static inline bool operator>(const unique<T> &lhs, const unique<T> &rhs) { return lhs.get() > rhs.get(); }
template<typename T> static inline bool operator>=(const unique<T> &lhs, const unique<T> &rhs) { return lhs.get() >= rhs.get(); }

} // namespace other
