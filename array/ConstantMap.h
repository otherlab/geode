//#####################################################################
// Class ConstantMap
//#####################################################################
#pragma once

#include <other/core/array/ArrayBase.h>
namespace other {

template<class T> struct IsArray<ConstantMap<T> >:public mpl::true_{};
template<class T> struct HasCheapCopy<ConstantMap<T> >:public mpl::true_{};

template<class T_>
class ConstantMap : public ArrayBase<T_,ConstantMap<T_> > {
  typedef T_ T;
public:
  typedef T Element;
private:
  int m;
  T constant;
public:

  ConstantMap(const int m,const T constant)
    : m(m), constant(constant)
  {}

  int size() const {
    return m;
  }

  const T& operator[](const int i) const {
    assert(unsigned(i)<unsigned(m));
    return constant;
  }
};

template<class T> static inline ConstantMap<T> constant_map(int n,const T& constant) {
  return ConstantMap<T>(n,constant);
}

}
