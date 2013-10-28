//#####################################################################
// Class Compute
//#####################################################################
#pragma once

#include <geode/value/Value.h>
#include <geode/value/Action.h>
#include <geode/structure/Tuple.h>
#include <geode/utility/curry.h>
#include <geode/utility/format.h>
#include <geode/utility/remove_const_reference.h>
#include <boost/function.hpp>
#include <stdio.h>
namespace geode {

using boost::function;

template<class T> class GEODE_CORE_CLASS_EXPORT Compute : public Value<T>,public Action {
public:
  GEODE_NEW_FRIEND
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

  vector<Ref<const ValueBase>> dependencies() const {
    return Action::dependencies();
  }
};

template<class F> static inline auto cache(const F& f)
  -> ValueRef<typename remove_const_reference<decltype(f())>::type> {
  typedef typename remove_const_reference<decltype(f())>::type T;
  return ValueRef<T>(new_<Compute<T>>(f));
}

#ifdef GEODE_VARIADIC

template<class A0,class A1,class... Args> static inline auto cache(const A0& a0, const A1& a1, const Args&... args)
  -> ValueRef<typename remove_const_reference<decltype(curry(a0,a1,args...)())>::type> {
  typedef typename remove_const_reference<decltype(curry(a0,a1,args...)())>::type T;
  return ValueRef<T>(new_<Compute<T>>(curry(a0,a1,args...)));
}

#else // Unpleasant nonvariadic versions

#define GEODE_CACHE(ARGS,Argsargs,args) \
  template<GEODE_REMOVE_PARENS(ARGS)> static inline auto cache Argsargs \
    -> ValueRef<typename remove_const_reference<decltype(curry args())>::type> { \
    typedef typename remove_const_reference<decltype(curry args())>::type T; \
    return ValueRef<T>(new_<Compute<T>>(curry args)); \
  }
GEODE_CACHE((class A0,class A1),(const A0& a0,const A1& a1),(a0,a1))
GEODE_CACHE((class A0,class A1,class A2),(const A0& a0,const A1& a1,const A2& a2),(a0,a1,a2))
GEODE_CACHE((class A0,class A1,class A2,class A3),(const A0& a0,const A1& a1,const A2& a2,const A3& a3),(a0,a1,a2,a3))
#undef GEODE_CACHE

#endif

}
