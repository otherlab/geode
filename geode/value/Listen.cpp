#include <geode/value/Listen.h>
#include <geode/utility/Log.h>
namespace geode {

using Log::cerr;

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
    save(e)->print(cerr,"Listen: exception in listener callback");
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
    save(e)->print(cerr,"Listen: exception in listener callback");
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
