#include <geode/value/ConstValue.h>
#include <geode/python/function.h>
#include <geode/python/Ptr.h>
#include <geode/python/wrap.h>
using namespace geode;

void wrap_const_value() {
  python::function("const_value_py",const_value<Ptr<>>);
}
