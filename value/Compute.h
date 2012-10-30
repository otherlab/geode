//#####################################################################
// Class Compute
//#####################################################################
#pragma once

#include <other/core/value/Value.h>
#include <other/core/value/Action.h>
#include <other/core/structure/Tuple.h>
#include <other/core/utility/curry.h>
#include <other/core/utility/format.h>
#include <other/core/utility/remove_const_reference.h>
#include <boost/function.hpp>
namespace other{

using boost::function;

template<class T> class Compute:public Value<T>,public Action
{
public:
  OTHER_NEW_FRIEND
  typedef Value<T> Base;
private:
  const function<T()> f;

protected:
  template<class F> Compute(const F& f)
    :f(f) {}
public:

  void input_changed() const {
    Base::set_dirty();
  }

  void update() const {
    Executing e(*this);
    this->set_value(e.stop(f())); // Note that e is stopped before set_value is called
  }

  void dump(int indent) const {
    printf("%*sCompute<%s>\n",2*indent,"",typeid(T).name());
    Action::dump_dependencies(indent);
  }
  std::vector<Ptr<const ValueBase> > get_dependencies() const {
    return Action::get_dependencies();
  }

};

template<class F> auto cache(const F& f)
  -> ValueRef<typename remove_const_reference<decltype(f())>::type> {
  typedef typename remove_const_reference<decltype(f())>::type T;
  return ValueRef<T>(new_<Compute<T>>(f));
}

template<class A0,class A1,class... Args> auto cache(const A0& a0, const A1& a1, const Args&... args)
  -> ValueRef<typename remove_const_reference<decltype(curry(a0,a1,args...)())>::type> {
  typedef typename remove_const_reference<decltype(curry(a0,a1,args...)())>::type T;
  return ValueRef<T>(new_<Compute<T>>(curry(a0,a1,args...)));
}

}
