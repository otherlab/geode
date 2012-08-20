#include "Listen.h"
#include <other/core/python/Class.h>
#include <other/core/python/module.h>
#include <other/core/python/function.h>

namespace other{

OTHER_DEFINE_TYPE(Listen)

Listen::Listen(Ref<const ValueBase> value, const function<void()>& f)
  : value(value)
  , f(f) {
  depend_on(*value);
}

Listen::~Listen() {}

void Listen::input_changed() const {
  Executing e; // register no dependencies during execution
  f();
  depend_on(*value);
}

}

using namespace other;

void wrap_listen(){
  typedef Listen Self;
  Class<Self>("Listen");
  python::function("listen",static_cast<Ref<Listen>(*)(const Ref<const ValueBase>&,const function<void()>&)>(&listen));
}

