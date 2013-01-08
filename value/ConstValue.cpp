#include <other/core/value/ConstValue.h>
#include <other/core/python/function.h>
#include <other/core/python/Ptr.h>
#include <other/core/python/wrap.h>
using namespace other;

void wrap_const_value() {
    python::function("const_value",const_value<Ptr<> >);
}
