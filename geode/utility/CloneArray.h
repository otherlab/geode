//#####################################################################
// Class CloneArray
//#####################################################################
#pragma once

#include <geode/utility/Cloneable.h>
#include <cassert>
namespace geode {

template<class T> class CloneArray;

template<>
class CloneArray<CloneableBase> {
private:
  const size_t sizeof_clone;
  const int count;
  char* data;
public:

  // Construct an array of default constructed type clones of the template object
  CloneArray(const CloneableBase& template_object, const int count);
  virtual ~CloneArray();

  int size() const {
    return count;
  }

  CloneableBase& operator()(const int i) {
    assert(unsigned(i)<unsigned(count));
    return *(CloneableBase*)(data+sizeof_clone*i);
  }

  const CloneableBase& operator()(const int i) const {
    assert(unsigned(i)<unsigned(count));
    return *(const CloneableBase*)(data+sizeof_clone*i);
  }
};

template<class T>
class CloneArray : public CloneArray<CloneableBase> {
  BOOST_MPL_ASSERT((IsCloneable<T>));
  typedef CloneArray<CloneableBase> Base;
public:
  CloneArray(const T& template_object, const int count)
    : CloneArray<CloneableBase>(template_object,count)
  {}

  T& operator()(const int i) {
    return static_cast<T&>(Base::operator()(i));
  }

  const T& operator()(const int i) const {
    return static_cast<const T&>(Base::operator()(i));
  }
};

}
