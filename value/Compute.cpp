#include <other/core/value/Compute.h>
#include <other/core/python/from_python.h>
#include <other/core/python/Ptr.h>
#include <other/core/python/Class.h>
#include <other/core/utility/format.h>
namespace other {
#ifdef OTHER_PYTHON

static PyObject* empty_tuple = 0;

namespace {
class ComputePython : public Value<Ptr<>>,public Action {
public:
  OTHER_DECLARE_TYPE(OTHER_NO_EXPORT)
  typedef ValueBase Base;

  const Ref<> f;

protected:
  ComputePython(PyObject* f)
    : f(ref(*f)) {}
public:

  void input_changed() const {
    this->set_dirty();
  }

  void update() const {
    Executing e(*this);
    this->set_value(e.stop(steal_ref_check(PyObject_Call(&*f,empty_tuple,0))));
  }

  void dump(int indent) const {
    printf("%*scache(%s)\n",2*indent,"",name().c_str());
    Action::dump_dependencies(indent);
  }

  vector<Ptr<const ValueBase>> get_dependencies() const {
    return Action::get_dependencies();
  }

  string name() const {
    return from_python<string>(python_field(f,"__name__"));
  }
};

OTHER_DEFINE_TYPE(ComputePython)
}

// We write a special version for python to allow easy introspection
static Ref<ValueBase> cache_py(PyObject* f) {
  if (!PyCallable_Check(f))
    throw TypeError("cache: argument is not callable");
  return new_<ComputePython>(f);
}

#endif
}
using namespace other;

void wrap_compute() {
#ifdef OTHER_PYTHON
  empty_tuple = PyTuple_New(0);
  OTHER_ASSERT(empty_tuple);

  typedef ComputePython Self; 
  Class<Self>("Compute")
    .OTHER_FIELD(f)
    .OTHER_GET(name)
    ;

  OTHER_FUNCTION_2(cache,cache_py)
#endif
}
