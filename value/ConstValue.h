//#####################################################################
// Class ConstValue
//#####################################################################
#pragma once

#include <other/core/value/Value.h>
namespace other{

template<class T> class ConstValue:public Value<T>
{
public:
  OTHER_NEW_FRIEND
  typedef Value<T> Base;
private:
  ConstValue(const T& value) {
    this->set_value(value);
  }

  void update() const {
    OTHER_FATAL_ERROR(); // We never go invalid, so this never be called
  }

  void dump(int indent) const {
    printf("%*sConstValue<%s>\n",2*indent,"",typeid(T).name());
  }

  std::vector<Ptr<const ValueBase> > get_dependencies() const {
    std::vector<Ptr<const ValueBase> > result;
    result.push_back(ptr(this));
    return result;
  }

};

template<class T> ValueRef<T> const_value(const T& value) {
  return ValueRef<T>(new_<ConstValue<T> >(value));
}

}
