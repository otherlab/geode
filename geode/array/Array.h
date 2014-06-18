//#####################################################################
// Class Array<T>
//#####################################################################
//
// Array represents a flat, one-dimensional array.  The array buffer can be shared between multiple instances
// without copying, since it is managed by an arbitrary PyObject.  By default, the owning PyObject is a Buffer,
// but if the array originated from python it could be a numpy array or other native PyObject.
//
// The const semantics of Array are analogous to pointers: const Array<T> can be modified but not resized,
// and Array<const T> can be resized but not modified.  These semantics are a consequence of shareability:
// const Array<T> can also be modified by copying it into an Array<T> first.
//
// Note that Array always owns its own data; owner is allowed to be null only for empty arrays.  RawArray
// should be used in threaded code to avoid reference counting for thread safety and speed, and is returned
// by slice and similar functions by default.
//
//#####################################################################
#pragma once

#include <geode/array/ArrayBase.h>
#include <geode/array/RawArray.h>
#include <geode/python/Buffer.h>
#include <geode/python/exceptions.h>
#include <geode/python/forward.h>
#include <geode/utility/debug.h>
#include <geode/utility/format.h>
#include <geode/utility/config.h>
#include <geode/utility/range.h>
#include <geode/utility/type_traits.h>
#include <geode/vector/Vector.h>
#include <geode/utility/const_cast.h>
#include <vector>
namespace geode {

using std::swap;

template<class T> struct IsArray<Array<T>>:public mpl::true_{};

// Check whether T is an array type with a shareable buffer
template<class TArray> struct IsShareableArray:public mpl::false_{};
template<class TArray> struct IsShareableArray<const TArray>:public IsShareableArray<TArray>{};

// Array<T> is shareable
template<class T> struct IsShareableArray<Array<T>>:public mpl::true_{};

// This cannot be GEODE_CORE_EXPORT, since it's defined as a template in headers
template<class T,int d> PyObject* to_python(const Array<T,d>& array);
template<class T,int d> struct FromPython<Array<T,d>>{static Array<T,d> convert(PyObject* object);};
template<class T,int d> struct has_to_python<Array<T,d>> : public has_to_python<T> {};
template<class T,int d> struct has_from_python<Array<T,d>> : public has_from_python<T> {};

template<class T_>
class Array<T_,1> : public ArrayBase<T_,Array<T_>> {
  typedef T_ T;
public:
  typedef typename remove_const<T>::type Element;
  static const bool is_const = geode::is_const<T>::value;
  typedef T& result_type;
  enum Workaround1 {dimension=1};
  enum Workaround2 {d=dimension};

  typedef ArrayBase<T,Array> Base;
  using Base::front;using Base::back;using Base::copy;using Base::same_array;
private:
  friend class Array<Element>;
  friend class Array<const Element>;
  struct Unusable{};

  int m_;
  int max_size_; // buffer size
  T* data_;
  PyObject* owner_; // python object that owns the buffer
public:

  Array()
    : m_(0), max_size_(0), data_(0), owner_(0) {}

  explicit Array(const int m_)
    : m_(m_), max_size_(m_) {
    assert(m_>=0);
    Buffer* buffer = Buffer::new_<T>(m_);
    data_ = (T*)buffer->data;
    owner_ = (PyObject*)buffer;
    if (IsScalarVectorSpace<T>::value)
      memset((void*)data_,0,m_*sizeof(T));
    else
      for (int i=0;i<m_;i++)
        const_cast<Element*>(data_)[i] = T();
  }

  explicit Array(const int m_, Uninit)
    : m_(m_), max_size_(m_) {
    assert(m_>=0);
    auto buffer = Buffer::new_<T>(m_);
    data_ = (T*)buffer->data;
    owner_ = (PyObject*)buffer;
  }

  explicit Array(const Vector<int,d> sizes)
    : Array(sizes.x) {}

  explicit Array(const Vector<int,d> sizes, Uninit)
    : Array(sizes.x,uninit) {}

  Array(const Array& source)
    : Base(), m_(source.m_), max_size_(source.max_size_), data_(source.data_), owner_(source.owner_) {
    assert(owner_ || !data_);
    // Share a reference to the source buffer without copying it
    GEODE_XINCREF(owner_);
  }

  Array(typename mpl::if_c<is_const,const Array<Element>&,Unusable>::type source)
    : m_(source.m_), max_size_(source.max_size_), data_(source.data_), owner_(source.owner_) {
    assert(owner_ || !data_);
    // Share a reference to the source buffer without copying it
    GEODE_XINCREF(owner_);
  }

  template<class TA>
  explicit Array(const TA& source, typename enable_if<IsShareableArray<TA>,Unusable>::type unused=Unusable())
    : m_(source.size()), max_size_(source.max_size_), data_(source.data_), owner_(source.owner_) {
    assert(owner_ || !data_);
    // Share a reference to the source buffer without copying it
    STATIC_ASSERT_SAME(Element,typename TA::Element);
    GEODE_XINCREF(owner_);
  }

  explicit Array(const NdArray<T>& array) {
    GEODE_ASSERT(array.rank()==1);
    m_ = max_size_=array.shape[0];
    data_ = array.data();
    owner_ = array.owner();
  }

  Array(const int m_, T* data, PyObject* owner)
    : m_(m_), max_size_(m_), data_(data), owner_(owner) {
    assert(owner_ || !data_);
    GEODE_XINCREF(owner_);
  }

  Array(const Vector<int,1> sizes, T* data, PyObject* owner)
    : Array(sizes.x,data,owner) {}

  ~Array() {
    GEODE_XDECREF(owner_);
  }

  RawArray<T> raw() const { // Return a non-owning array for use in threaded code where reference counting is bad
    return RawArray<T>(m_,data_);
  }

  int size() const {
    return m_;
  }

  int total_size() const {
    return m_;
  }

  Vector<int,1> sizes() const {
    return Vector<int,1>(m_);
  }

  T& operator[](const int i) const {
    assert(unsigned(i)<unsigned(m_));
    return data_[i];
  }

  T& operator()(const int i) const {
    assert(unsigned(i)<unsigned(m_));
    return data_[i];
  }

  bool valid(const int i) const {
    return unsigned(i)<unsigned(m_);
  }

  T* data() const {
    return data_;
  }

  PyObject* owner() const {
    GEODE_XINCREF(owner_);
    return owner_;
  }

  PyObject* borrow_owner() const {
    return owner_;
  }

  int max_size() const {
    return max_size_;
  }

  void clean_memory() {
    Array empty;
    swap(empty);
  }

  void clear() {
    m_ = 0;
  }

  void swap(Array& other) {
    std::swap(m_,other.m_);
    std::swap(max_size_,other.max_size_);
    std::swap(data_,other.data_);
    std::swap(owner_,other.owner_);
  }

  Array& operator=(const Array& source) {
    PyObject* owner_save = owner_;
    // Share a reference to the source buffer without copying it
    GEODE_XINCREF(source.owner_);
    owner_ = source.owner_;
    m_ = source.m_;
    max_size_ = source.max_size_;
    data_ = source.data_;
    // Call decref last in case of side effects or this==&source
    GEODE_XDECREF(owner_save);
    return *this;
  }

  template<class TArray> typename enable_if<IsShareableArray<TArray>,Array&>::type operator=(const TArray& source) {
    assert(source.owner_ || !source.data_);
    PyObject* owner_save = owner_;
    // Share a reference to the source buffer without copying it
    GEODE_XINCREF(source.owner_);
    owner_ = source.owner_;
    m_ = source.m_;
    max_size_ = source.max_size_;
    data_ = source.data_;
    // Call decref last in case of side effects or this==&source
    GEODE_XDECREF(owner_save);
    return *this;
  }

  template<class TArray> void copy(const TArray& source) {
    // Copy data from source array even if it is shareable
    STATIC_ASSERT_SAME(T,typename TArray::value_type);
    const int source_m = source.size();
    m_ = 0;
    if (max_size_<source_m)
      grow_buffer(source_m);
    if (!same_array(*this,source))
      for (int i=0;i<source_m;i++)
        data_[i] = source[i];
    m_ = source_m;
  }

  template<class TArray> void copy(const TArray& source) const {
    // Const, so no resizing allowed
    STATIC_ASSERT_SAME(T,typename TArray::value_type);
    const int source_m = source.size();
    assert(m_==source_m);
    if (!same_array(*this,source))
      for (int i=0;i<source_m;i++)
        data_[i] = source[i];
  }

private:
  void grow_buffer(const int max_size_new) {
    if (max_size_>=max_size_new) return;
    Buffer* new_owner = Buffer::new_<T>(max_size_new);
    const int m_ = this->m_; // teach compiler that m_ is constant
    for (int i=0;i<m_;i++)
      ((typename remove_const<T>::type*)new_owner->data)[i] = data_[i];
    GEODE_XDECREF(owner_);
    max_size_ = max_size_new;
    data_ = (T*)new_owner->data;
    owner_ = (PyObject*)new_owner;
  }
public:

  void preallocate(const int m_new) GEODE_ALWAYS_INLINE {
    if(max_size_<m_new)
      grow_buffer(geode::max(4*max_size_/3+2,m_new));
  }

  void resize(const int m_new) {
    preallocate(m_new);
    if (m_new>m_) {
      if (IsScalarVectorSpace<T>::value)
        memset((void*)(data_+m_),0,(m_new-m_)*sizeof(T));
      else
        for (int i=m_;i<m_new;i++) data_[i] = T();
    }
    m_ = m_new;
  }

  void resize(const int m_new, Uninit) GEODE_ALWAYS_INLINE {
    preallocate(m_new);
    m_ = m_new;
  }

  void exact_resize(const int m_new) { // Zero elbow room
    if (m_==m_new) return;
    int m_end = geode::min(m_,m_new);
    if (max_size_!=m_new) {
      Buffer* new_owner = Buffer::new_<T>(m_new);
      for (int i=0;i<m_end;i++)
        new_owner->data[i] = data_[i];
      GEODE_XDECREF(owner_);
      max_size_ = m_new;
      data_ = (T*)new_owner->data;
      owner_ = (PyObject*)new_owner;
    }
    if (m_new>m_end) {
      if (IsScalarVectorSpace<T>::value)
        memset((void*)(data_+m_end),0,(m_new-m_end)*sizeof(T));
      else
        for (int i=m_end;i<m_new;i++)
          data_[i] = T();
    }
    m_ = m_new;
  }

  void exact_resize(const int m_new, Uninit) { // Zero elbow room
    if (m_==m_new) return;
    int m_end = geode::min(m_,m_new);
    if (max_size_!=m_new) {
      Buffer* new_owner = Buffer::new_<T>(m_new);
      for (int i=0;i<m_end;i++)
        new_owner->data[i] = data_[i];
      GEODE_XDECREF(owner_);
      max_size_ = m_new;
      data_ = (T*)new_owner->data;
      owner_ = (PyObject*)new_owner;
    }
    m_ = m_new;
  }

  void compact() { // Note: if the buffer is shared, the memory will not be deallocated
    if (m_<max_size_)
      exact_resize(m_);
  }

  RawArray<T> reshape(int m_new) const {
    assert(m_new==m_);
    return RawArray<T>(m_new,data());
  }

  const Array<T>& reshape_own(int m_new) const {
    assert(m_new==m_);
    return *this;
  }

  RawArray<T,2> reshape(int m_new,int n_new) const {
    assert(m_new*n_new==m_);
    return RawArray<T,2>(m_new,n_new,data());
  }

  const Array<T,2> reshape_own(int m_new,int n_new) const {
    assert(m_new*n_new==m_);
    return Array<T,2>(m_new,n_new,data(),owner_);
  }

  const Array<T,2> reshape_own(Vector<int,2> new_sizes) const {
    return reshape_own(new_sizes.x,new_sizes.y);
  }

  RawArray<T,3> reshape(int m_new,int n_new,int mn_new) const {
    assert(m_new*n_new*mn_new==m_);
    return RawArray<T,3>(m_new,n_new,mn_new,data());
  }

  const Array<T,3> reshape_own(int m_new,int n_new,int mn_new) const {
    assert(m_new*n_new*mn_new==m_);
    return Array<T,3>(m_new,n_new,mn_new,data(),owner_);
  }

  const Array<T,3> reshape_own(Vector<int,3> new_sizes) const {
    return reshape_own(new_sizes.x,new_sizes.y,new_sizes.z);
  }

  int append(const T& element) GEODE_ALWAYS_INLINE {
    if (m_<max_size_)
      data_[m_++] = element;
    else {
      T save = element; // element could be reference into the current array
      preallocate(m_+1);
      data_[m_++] = save;
    }
    return m_-1;
  }

  int append(Uninit) GEODE_ALWAYS_INLINE {
    preallocate(m_+1);
    return m_++;
  }

  int append_assuming_enough_space(const T& element) GEODE_ALWAYS_INLINE {
    assert(m_<max_size_);
    data_[m_++] = element;
    return m_-1;
  }

  template<class TArray> void extend(const TArray& append_array) {
    STATIC_ASSERT_SAME(typename remove_const<T>::type,typename remove_const<typename TArray::value_type>::type);
    int append_m = append_array.size(),
        m_new = m_+append_m;
    preallocate(m_new);
    for (int i=0;i<append_m;i++)
      geode::const_cast_(data_[m_+i]) = append_array[i];
    m_ = m_new;
  }

  int extend(const int n, Uninit) GEODE_ALWAYS_INLINE {
    const int m_old = m_,
              m_new = m_+n;
    preallocate(m_new);
    m_ = m_new;
    return m_old;
  }

  bool is_unique() const {
    Array a;
    a.append_unique_elements(*this);
    return a.size() == size();
  }

  void append_unique(const T& element) {
    if (!Base::contains(element))
      append(element);
  }

  template<class TArray> void append_unique_elements(const TArray& append_array) {
    STATIC_ASSERT_SAME(T,typename TArray::value_type);
    int append_m = append_array.size();
    for (int i=0;i<append_m;i++)
      append_unique(append_array(i));
  }

  void remove_index(const int index) { // Preserves ordering of remaining elements
    assert(unsigned(index)<unsigned(m_));
    for (int i=index;i<m_-1;i++)
      data_[i] = data_[i+1];
    m_--;
  }

  void remove_index_lazy(const int index) { // Fill holes with back()
    assert(unsigned(index)<unsigned(m_));
    data_[index] = data_[--m_];
  }

  void remove_first_lazy(T const &k) {
    int idx = Base::find(k);
    if (idx != -1)
      remove_index_lazy(idx);
  }

  void insert(const T& element, const int index) {
    preallocate(m_+1);
    m_++;
    for (int i=m_-1;i>index;i--)
      data_[i] = data_[i-1];
    data_[index] = element;
  }

  T& pop() { // Returns a temporarily valid reference (safe since ~T() is trivial)
    assert(m_);
    return data_[--m_];
  }

  Array<const T> pop_elements(const int count) { // Return value shares ownership with original
    static_assert(has_trivial_destructor<T>::value,"");
    assert(m_-count>=0);
    m_ -= count;
    return Array<const T>(count,data_+m_,owner_);
  }

  Array<Element>& const_cast_() {
    return *(Array<Element>*)this;
  }

  const Array<Element>& const_cast_() const {
    return *(const Array<Element>*)this;
  }

  const Array<const Element>& const_() const {
    return *(const Array<const Element>*)this;
  }

  RawArray<T> slice(int lo,int hi) const {
    assert(unsigned(lo)<=unsigned(hi) && unsigned(hi)<=unsigned(m_));
    return RawArray<T>(hi-lo,data_+lo);
  }

  Array<T> slice_own(int lo,int hi) const {
    assert(unsigned(lo)<=unsigned(hi) && unsigned(hi)<=unsigned(m_));
    return Array(hi-lo,data_+lo,owner_);
  }

  RawArray<T> slice(Range<int> range) const {
    return slice(range.lo,range.hi);
  }

  Array<T> slice_own(Range<int> range) const {
    return slice_own(range.lo,range.hi);
  }

  void zero() const {
    static_assert(IsScalarVectorSpace<T>::value,"");
    memset((void*)data_,0,m_*sizeof(T));
  }

  template<class T2> typename enable_if<is_same<T2,Element>,Array<T>>::type as() const {
    return *this;
  }

  template<class T2> typename disable_if<is_same<T2,Element>,Array<T2>>::type as() const {
    Array<typename remove_const<T2>::type> copy(m_,uninit);
    for (int i=0;i<m_;i++) copy[i] = T2(data_[i]);
    return copy;
  }
};

template<class T,int d>   static inline const RawArray<T>       asarray(T (&v)[d])                 { return RawArray<T>(d,v); }
template<class T,int d>   static inline const RawArray<T>       asarray(Vector<T,d>& v)            { return RawArray<T>(d,v.begin()); }
template<class T,int d>   static inline const RawArray<const T> asarray(const Vector<T,d>& v)      { return RawArray<const T>(d,v.begin()); }
template<class T>         static inline const RawArray<T>&      asarray(const RawArray<T>& v)      { return v; }
template<class T>         static inline const RawArray<T>       asarray(const Array<T>& v)         { return v; }
template<class T,class A> static inline const RawArray<T>       asarray(std::vector<T,A>& v)       { assert(v.size() <= std::numeric_limits<int>::max()); return RawArray<T>(int(v.size()),&v[0]); }
template<class T,class A> static inline const RawArray<const T> asarray(const std::vector<T,A>& v) { assert(v.size() <= std::numeric_limits<int>::max()); return RawArray<const T>(int(v.size()),&v[0]); }
template<class T,class A> static inline const A&                asarray(const ArrayBase<T,A>& v)   { return v.derived(); }

template<class T,int d>   static inline const RawArray<const T> asconstarray(T (&v)[d])                 { return RawArray<const T>(d,v); }
template<class T,int d>   static inline const RawArray<const T> asconstarray(const Vector<T,d>& v)      { return RawArray<const T>(d,v.begin()); }
template<class T>         static inline const RawArray<const T> asconstarray(const RawArray<T>& v)      { return v; }
template<class T>         static inline const RawArray<const T> asconstarray(const Array<T>& v)         { return v; }
template<class T,class A> static inline const RawArray<const T> asconstarray(const std::vector<T,A>& v) { return RawArray<const T>(v.size(),&v[0]); }
template<class T,class A> static inline const A&                asconstarray(const ArrayBase<T,A>& v)   { return v.derived(); }

template<class T,int d> static inline       Array<T>& flat(      Array<T,d>& A) { return A.flat; }
template<class T>       static inline       Array<T>& flat(      Array<T,1>& A) { return A; }
template<class T,int d> static inline const Array<T>& flat(const Array<T,d>& A) { return A.flat; }
template<class T>       static inline const Array<T>& flat(const Array<T,1>& A) { return A; }

}
namespace std{
template<class T,int d> void swap(geode::Array<T,d>& array1, geode::Array<T,d>& array2) {
  array1.swap(array2);
}
}
