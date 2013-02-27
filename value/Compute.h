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
#include <stdio.h>
namespace other {

using boost::function;

template<class T> class OTHER_CORE_CLASS_EXPORT Compute : public Value<T>,public Action {
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

  vector<Ptr<const ValueBase>> get_dependencies() const {
    return Action::get_dependencies();
  }
};

template<class F> static inline auto cache(const F& f)
  -> ValueRef<typename remove_const_reference<decltype(f())>::type> {
  typedef typename remove_const_reference<decltype(f())>::type T;
  return ValueRef<T>(new_<Compute<T>>(f));
}

#ifdef OTHER_VARIADIC

template<class A0,class A1,class... Args> static inline auto cache(const A0& a0, const A1& a1, const Args&... args)
  -> ValueRef<typename remove_const_reference<decltype(curry(a0,a1,args...)())>::type> {
  typedef typename remove_const_reference<decltype(curry(a0,a1,args...)())>::type T;
  return ValueRef<T>(new_<Compute<T>>(curry(a0,a1,args...)));
}

#else // Unpleasant nonvariadic versions

#define OTHER_CACHE(ARGS,Argsargs,args) \
  template<OTHER_REMOVE_PARENS(ARGS)> static inline auto cache Argsargs \
    -> ValueRef<typename remove_const_reference<decltype(curry args())>::type> { \
    typedef typename remove_const_reference<decltype(curry args())>::type T; \
    return ValueRef<T>(new_<Compute<T>>(curry args)); \
  }
OTHER_CACHE((class A0,class A1),(const A0& a0,const A1& a1),(a0,a1))
OTHER_CACHE((class A0,class A1,class A2),(const A0& a0,const A1& a1,const A2& a2),(a0,a1,a2))
OTHER_CACHE((class A0,class A1,class A2,class A3),(const A0& a0,const A1& a1,const A2& a2,const A3& a3),(a0,a1,a2,a3))
#undef OTHER_CACHE

#endif

}
