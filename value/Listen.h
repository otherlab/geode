//#####################################################################
// Class Listen
//#####################################################################
#pragma once

#include <other/core/value/Value.h>
#include <other/core/value/Action.h>
#include <boost/function.hpp>
namespace other {

using boost::function;

class Listen : public Object, public Action {
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
  typedef Object Base;
private:
  Ref<const ValueBase> value;
  function<void()> f;

  OTHER_CORE_EXPORT Listen(const ValueBase& value, const function<void()>& f);
public:
  ~Listen();

  void input_changed() const;
};


class BatchListen : public Object, public Action {
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
  typedef Object Base;
private:
  vector<Ref<const ValueBase>> values;
  function<void()> f;

  OTHER_CORE_EXPORT BatchListen(const vector<Ref<const ValueBase>>& values, const function<void()>& f);
public:
  ~BatchListen();

  void input_changed() const;
};

OTHER_CORE_EXPORT Ref<Listen> listen(const ValueBase& value, const function<void()>& f);
OTHER_CORE_EXPORT Ref<BatchListen> batch_listen(const vector<Ref<const ValueBase>>& values, const function<void()>& f);

}
