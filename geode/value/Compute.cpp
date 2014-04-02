#include <geode/value/Compute.h>
#include <geode/utility/Ptr.h>
#include <geode/utility/format.h>
namespace geode {
#if 0 // Value python support

static PyObject* empty_tuple = 0;

namespace {
class CachePython : public Value<Ptr<>>,public Action {
public:
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT)
  typedef ValueBase Base;

  const Ref<> f;
  PyObject *_name;

protected:
  CachePython(PyObject* f)
    : f(ref(*f)), _name(NULL) {}

  CachePython(PyObject* f, string const &name)
    : f(ref(*f)), _name(to_python(name)) {}
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

  vector<Ref<const ValueBase>> dependencies() const {
    return Action::dependencies();
  }

  string name() const {
    if (_name)
      return from_python<string>(_name);
    else
      return from_python<string>(python_field(f,"__name__"));
  }

  string repr() const {
    return format("cache(%s)",name());
  }
};

GEODE_DEFINE_TYPE(CachePython)
}

// We write a special version for python to allow easy introspection
static Ref<ValueBase> cache_py(PyObject* f) {
  if (!PyCallable_Check(f))
    throw TypeError("cache: argument is not callable");
  return new_<CachePython>(f);
}

// this decorator takes a name instead of extracting it from the callable
static Ref<ValueBase> cache_named_inner(PyObject* f, string const &name) {
  if (!PyCallable_Check(f))
    throw TypeError("cache: argument is not callable");
  return new_<CachePython>(f, name);
}

#endif
}
