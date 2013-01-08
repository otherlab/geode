#include <other/core/value/Compute.h>
#include <other/core/python/from_python.h>
#include <other/core/python/function.h>
#include <other/core/python/Ptr.h>
#include <other/core/python/wrap.h>
#include <other/core/utility/format.h>
using namespace other;

void wrap_compute() {
  python::function("cache",static_cast<ValueRef<Ptr<> >(*)(const function<Ptr<>()>&)>(cache));
}
