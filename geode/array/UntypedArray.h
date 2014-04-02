// An array with no type information
#pragma once

#include <geode/array/Array.h>
namespace geode {

// An UntypedArray has a prefix which is binary compatible with Array<T> and Field<T,Id>,
// but which does not know its own type.  Once the type is known, it can be safely cast to Array<T>.

class UntypedArray {
  // This part is ABI compatible with Array<T>
  int m_;                         // Size in elements
  int max_size_;                  // Buffer size in elements
  char* data_;                    // max_size_*t_size_ bytes
  shared_ptr<const Owner> owner_; // Object that owns the buffer

  // Type information
  int t_size_;
  const type_info* type_;

  struct Copy {};
public:

  UntypedArray(const type_info* type, int t_size)
    : m_(0)
    , max_size_(0)
    , data_(0)
    , t_size_(t_size)
    , type_(type)
  {}

  // Create an empty untyped array
  template<class T> UntypedArray(Types<T>)
    : UntypedArray(&typeid(T), sizeof(T)) {
    static_assert(has_trivial_destructor<T>::value,"UntypedArray can only store POD-like types");
  }

  // Create an initialized (zeroed) untyped array
  template<class T> UntypedArray(Types<T> t, const int size)
    : UntypedArray(t,size,uninit) {
    memset(data_,0,t_size_*m_);
  }

  // Create an uninitialized untyped array
  template<class T> UntypedArray(Types<T>, const int size, Uninit)
    : m_(size)
    , max_size_(size)
    , t_size_(sizeof(T))
    , type_(&typeid(T)) {
    static_assert(has_trivial_destructor<T>::value,"UntypedArray can only store POD-like types");
    const auto buffer = Buffer::new_<T>(m_);
    data_ = buffer->data;
    owner_ = buffer;
  }

  // Share ownership with an untyped array
  UntypedArray(const UntypedArray& o)
    : m_(o.m_)
    , max_size_(o.max_size_)
    , data_(o.data_)
    , owner_(o.owner_)
    , t_size_(o.t_size_)
    , type_(o.type_) {
    assert(owner_ || !data_);
  }

  // Copy an UntypedArray.  Use via copy() below
  UntypedArray(const UntypedArray& o, Copy)
    : m_(o.m_)
    , max_size_(o.m_)
    , t_size_(o.t_size_)
    , type_(o.type_) {
    const auto buffer = Buffer::new_<char>(m_*t_size_);
    data_ = buffer->data;
    owner_ = buffer;
    memcpy(data_,o.data_,m_*t_size_);
  }

  // Share ownership with an input field
  template<class T,class Id> UntypedArray(const Field<T,Id>& f)
    : m_(f.size())
    , max_size_(f.flat.max_size())
    , data_((char*)f.flat.data())
    , owner_(f.flat.owner())
    , t_size_(sizeof(T))
    , type_(&typeid(T))
  {
    static_assert(has_trivial_destructor<T>::value,"UntypedArray can only store POD-like types");
    assert(owner_ || !data_);
  }

  UntypedArray& operator=(const UntypedArray& o) {
    owner_ = o.owner_;
    m_ = o.m_;
    max_size_ = o.max_size_;
    data_ = o.data_;
    t_size_ = o.t_size_;
    type_ = o.type_;
    return *this;
  }

  // Copy all aspects of an UntypedArray, except give it a new size (and don't copy any data)
  static UntypedArray empty_like(const UntypedArray &o, int new_size) {
    UntypedArray A(o.type_, o.t_size_);
    A.resize(new_size, false, false);
    return A;
  }

  int size() const {
    return m_;
  }

  int t_size() const {
    return t_size_;
  }

  const type_info& type() const {
    return *type_;
  }

  char* data() const {
    return data_;
  }

  UntypedArray copy() const {
    return UntypedArray(*this,Copy());
  }

private:
  void grow_buffer(const int max_size_new, const bool copy_existing=true) {
    if (max_size_ >= max_size_new)
      return;
    const auto new_owner = Buffer::new_<char>(max_size_new*t_size_);
    if (copy_existing)
      memcpy(new_owner->data,data_,t_size_*m_);
    max_size_ = max_size_new;
    data_ = new_owner->data;
    owner_ = new_owner;
  }
public:

  void preallocate(const int m_new, const bool copy_existing=true) GEODE_ALWAYS_INLINE {
    if (max_size_ < m_new)
      grow_buffer(geode::max(4*max_size_/3+2,m_new),copy_existing);
  }

  void resize(const int m_new, const bool initialize_new=true, const bool copy_existing=true) {
    preallocate(m_new,copy_existing);
    if (initialize_new && m_new>m_)
      memset(data_+m_*t_size_,0,(m_new-m_)*t_size_);
    m_ = m_new;
  }

  void extend(const int extra) {
    resize(m_+extra);
  }

  void extend(const UntypedArray& o) {
    GEODE_ASSERT(t_size_ == o.t_size_);
    const int om = o.m_, m = m_;
    preallocate(m+om);
    memcpy(data_+m*t_size_,o.data_,om*t_size_);
    m_ += om;
  }

  void zero(const int i) const {
    assert(unsigned(i)<unsigned(m_));
    memset(data_+i*t_size_,0,t_size_);
  }

  void swap(const int i, const int j) const {
    assert(unsigned(i)<unsigned(m_) && unsigned(j)<unsigned(m_));
    char *p = data_+i*t_size_,
         *q = data_+j*t_size_;
    for (int k=0;k<t_size_;k++)
      swap(p[k],q[k]);
  }

  void copy(int to, int from) {
    memcpy(data_+to*t_size_,data_+from*t_size_,t_size_);
  }

  // copy o[j] to this[i]
  void copy_from(int i, UntypedArray const &o, int j) {
    // only allowed if types are the same
    assert(type_ == o.type_);
    memcpy(data_+i*t_size_,o.data_+j*t_size_,t_size_);
  }

  // Typed access to data

  template<class T> T& get(const int i) const {
    assert(sizeof(T)==t_size_ && unsigned(i)<unsigned(m_));
    return ((T*)data_)[i];
  }

  template<class T> int append(const T& x) {
    extend(1);
    const int i = m_-1;
    get<T>(i) = x;
    return i;
  }

  template<class T> const Array<T>& get() const {
    GEODE_ASSERT(*type_ == typeid(T));
    return *(Array<T>*)this;
  }

  template<class T,class Id> const Field<T,Id>& get() const {
    GEODE_ASSERT(*type_ == typeid(T));
    return *(Field<T,Id>*)this;
  }

  // cast to another type of the same length
  template<class T> const Array<T>& cast_get() const {
    GEODE_ASSERT(sizeof(T) == t_size_);
    return *(Array<T>*)this;
  }

  // cast to another type of the same length
  template<class T,class Id> const Field<T,Id>& cast_get() const {
    GEODE_ASSERT(sizeof(T) == t_size_);
    return *(Field<T,Id>*)this;
  }
};

}
