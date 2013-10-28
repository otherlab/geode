// Link two properties together: when one changes, the other will
#pragma once

#include <geode/value/Prop.h>
#include <geode/value/Listen.h>
namespace geode {

class PropLink : public Object {
public:
  GEODE_NEW_FRIEND
  typedef Object Base;

private:
  const Ref<> x,y,lx,ly;

  template<class T> PropLink(const PropRef<T>& x, const PropRef<T>& y)
    : x(x.self)
    , y(y.self)
    , lx(listen(x,curry(&PropLink::update<T>,&*y,&*x)))
    , ly(listen(y,curry(&PropLink::update<T>,&*x,&*y))) {}

  template<class T> static void update(Prop<T>* dst, Prop<T>* src) {
    dst->set((*src)());
  }
};

template<class T> Ref<> link(const PropRef<T>& x, const PropRef<T>& y) {
  return new_<PropLink>(x,y);
}

}
