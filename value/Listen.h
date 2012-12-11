//#####################################################################
// Class Listen
//#####################################################################
#pragma once

#include <other/core/value/Value.h>
#include <other/core/value/Action.h>
#include <other/core/utility/format.h>
#include <boost/function.hpp>
namespace other {

using boost::function;

class Listen: public Object, public Action
{
public:
  OTHER_DECLARE_TYPE
  typedef Object Base;
private:
  Ref<const ValueBase> value;
  function<void()> f;

  OTHER_CORE_EXPORT Listen(Ref<const ValueBase> value, const function<void()>& f) ;

public:
  ~Listen();

  void input_changed() const;
};

inline Ref<Listen> listen(const Ref<const ValueBase>& value, const function<void()>& f) {
  return new_<Listen>(value,f);
}

template<class T> Ref<Listen> listen(const PropRef<T>& value, const function<void()>& f) {
  return listen(value.self,f);
}

template<class T> Ref<Listen> listen(const ValueRef<T>& value, const function<void()>& f) {
  return listen(value.self,f);
}

}
