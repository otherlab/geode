//#####################################################################
// Class ConstValue
//#####################################################################
#pragma once

#include <geode/value/Value.h>
#include <stdio.h>
namespace geode {

template<class T> class ConstValue:public Value<T>
{
public:
  GEODE_NEW_FRIEND
  typedef Value<T> Base;
private:
  ConstValue(const T& value) {
    this->set_value(value);
  }

  void update() const {
    GEODE_FATAL_ERROR(); // We never go invalid, so this should never be called
  }

  void dump(int indent) const {
    printf("%*sConstValue<%s>\n",2*indent,"",typeid(T).name());
  }

  vector<Ref<const ValueBase>> dependencies() const {
    return vector<Ref<const ValueBase>>();
  }
};

template<class T> ValueRef<T> const_value(const T& value) {
  return ValueRef<T>(new_<ConstValue<T> >(value));
}

}
