//#####################################################################
// Class Listen
//#####################################################################
#pragma once

#include <geode/value/Value.h>
#include <geode/value/Action.h>
#include <geode/utility/function.h>
namespace geode {

class Listen : public Object, public Action {
public:
  GEODE_NEW_FRIEND
  typedef Object Base;
private:
  const Ref<const ValueBase> value;
  const function<void()> f;

  GEODE_CORE_EXPORT Listen(const ValueBase& value, const function<void()>& f);
public:
  ~Listen();

  void input_changed() const;
};


class BatchListen : public Object, public Action {
public:
  GEODE_NEW_FRIEND
  typedef Object Base;
private:
  const vector<Ref<const ValueBase>> values;
  const function<void()> f;

  GEODE_CORE_EXPORT BatchListen(const vector<Ref<const ValueBase>>& values, const function<void()>& f);
public:
  ~BatchListen();

  void input_changed() const;
};

GEODE_CORE_EXPORT Ref<Listen> listen(const ValueBase& value, const function<void()>& f);
GEODE_CORE_EXPORT Ref<BatchListen> batch_listen(const vector<Ref<const ValueBase>>& values, const function<void()>& f);

}
