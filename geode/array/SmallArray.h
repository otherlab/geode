#pragma once
#include <geode/array/ArrayBase.h>

namespace geode {


template<class T, int max_d> class SmallArray;

template<class T, int d> struct IsArray<SmallArray<T,d>>:public mpl::true_{};

template<class T_, int max_size_> class SmallArray : public ArrayBase<T_,SmallArray<T_, max_size_>> {
  typedef T_ T;
public:
  typedef typename remove_const<T>::type Element;
  static const bool is_const = geode::is_const<T>::value;

private:
  int m_;

  // We store data in a fixed size buffer
  // This is mutable since a 'const ArrayBase' still expects to be able to get non-const reference to elements as long as T isn't const
  mutable Vector<T, max_size_> buffer;
public:

  SmallArray()
   : m_(0) { }

  explicit SmallArray(int n, Uninit);

  explicit SmallArray(const Vector<T, max_size_>& values)
   : m_(values.size())
   , buffer(values)
  { }

  int size() const {
    assert(m_ <= max_size_);
    return m_;
  }

  T* data() const {
    return &buffer[0];
  }

  T& operator[](const int i) const {
    assert(unsigned(i)<unsigned(m_));
    return data()[i];
  }

  T& operator()(const int i) const {
    assert(unsigned(i)<unsigned(m_));
    return data()[i];
  }

  bool valid(const int i) const {
    return unsigned(i)<unsigned(m_);
  }

  static constexpr int max_size() {
    return max_size_;
  }

  void clear() {
    m_ = 0;
  }

  void resize(const int m_new) {
    assert(m_new <= max_size_);
    if (m_new > m_) {
      if (IsScalarVectorSpace<T>::value)
        memset((void*)(data()+m_),0,(m_new-m_)*sizeof(T));
      else
        for (int i=m_;i<m_new;i++) data()[i] = T();
    }
    m_ = m_new;
  }

  void resize(const int m_new, Uninit) GEODE_ALWAYS_INLINE {
    assert(m_new <= max_size_);
    m_ = m_new;
  }

  int append(Uninit) GEODE_ALWAYS_INLINE {
    assert(m_ < max_size_);
    return m_++;
  }

  int append(const T& element) GEODE_ALWAYS_INLINE {
    assert(m_ < max_size_);
    data()[m_++] = element;
    return m_-1;
  }

  T& pop() {
    assert(m_);
    return data()[--m_];
  }

};

} // namespace geode