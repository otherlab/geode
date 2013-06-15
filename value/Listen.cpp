#include "Listen.h"
#include <other/core/python/Class.h>
#include <other/core/python/function.h>
#include <other/core/python/wrap.h>
namespace other {

OTHER_DEFINE_TYPE(Listen)
OTHER_DEFINE_TYPE(BatchListen)

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

Ref<BatchListen> listen(const vector<Ref<const ValueBase>>& values, const function<void()>& f) {
  return new_<BatchListen>(values,f);
}

}
using namespace other;

void wrap_listen() {
  typedef Listen Self;
  Class<Self>("Listen");
  OTHER_FUNCTION_2(listen, static_cast<Ref<Listen>(*)(const ValueBase&,const function<void()>&)>(listen))
}
