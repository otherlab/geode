#include "Listen.h"
#include <geode/python/Class.h>
#include <geode/python/function.h>
#include <geode/python/wrap.h>
namespace geode {

GEODE_DEFINE_TYPE(Listen)
GEODE_DEFINE_TYPE(BatchListen)

Listen::Listen(const ValueBase& value, const function<void()>& f)
  : value(ref(value))
  , f(f) {
  depend_on(value);
}

Listen::~Listen() {}

void Listen::input_changed() const {
  try {
    Executing e; // register no dependencies during execution
    f();
  } catch (const exception& e) {
    print_and_clear_exception("Listen: exception in listener callback",e);
  }
  depend_on(*value);
}

BatchListen::BatchListen(const vector<Ref<const ValueBase>>& vv, const function<void()>& f)
  : values(vv)
  , f(f) {
  for(const auto v : values)
    depend_on(v);
}

BatchListen::~BatchListen() {}

void BatchListen::input_changed() const {
  try {
    Executing e; // register no dependencies during execution
    f();
  } catch (const exception& e) {
    print_and_clear_exception("Listen: exception in listener callback",e);
  }
  for(const auto v : values)
    depend_on(*v);
}

Ref<Listen> listen(const ValueBase& value, const function<void()>& f) {
  return new_<Listen>(value,f);
}

Ref<BatchListen> batch_listen(const vector<Ref<const ValueBase>>& values, const function<void()>& f) {
  return new_<BatchListen>(values,f);
}

}
using namespace geode;

void wrap_listen() {
  typedef Listen Self;
  Class<Self>("Listen");
  GEODE_FUNCTION_2(listen, static_cast<Ref<Listen>(*)(const ValueBase&,const function<void()>&)>(listen))
}
