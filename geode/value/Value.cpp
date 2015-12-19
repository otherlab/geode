#include <geode/array/Array.h>
#include <geode/value/Value.h>
#include <geode/value/Action.h>
#include <geode/value/Prop.h>
#include <geode/value/convert.h>
#include <geode/python/Class.h>
#include <geode/python/stl.h>
#include <geode/python/wrap.h>
#include <geode/utility/const_cast.h>
#include <geode/python/Ref.h>
#include <geode/utility/Hasher.h>
#include <iostream>
namespace geode {

using std::cout;
using std::cerr;
using std::endl;
using std::exception;

GEODE_DEFINE_TYPE(ValueBase)

// Link list of actions which have pending signals.  The fact that this is a two-sided doubly-linked list
// is important, since Actions need to be able to delete their links from the pending list if they
// destruct during dependency propagation.
GEODE_THREAD_LOCAL ValueBase::Link* ValueBase::pending = 0;

ValueBase::ValueBase(string const &s)
  : dirty_(true), name_(s), actions(0)
{}

ValueBase::~ValueBase() {
  // Unlink ourselves from all actions
  Link* link = actions;
  while (link) {
    Link* next = link->value_next;
    // Remove from the action's linked list
    *link->action_prev = link->action_next;
    if (link->action_next)
      link->action_next->action_prev = link->action_prev;
    delete link;
    link = next;
  }
}

bool ValueBase::is_type(const type_info& type) const {
  const type_info& self = this->type();
  // Use string comparison to avoid symbol visibility issues
  return type==self || !strcmp(type.name(),self.name());
}

inline void ValueBase::signal_pending() {
  while (pending) {
    Link* next = pending->value_next;
    // Remove from the pending linked list
    if (next)
      next->value_prev = &pending;
    // Remove from the action's linked list
    *pending->action_prev = pending->action_next;
    if (pending->action_next)
      pending->action_next->action_prev = pending->action_prev;
    // Delete link
    Action* action = pending->action;
    delete pending;
    pending = next;
    // Signal action
    if (!action->executing) {
      try {
        action->input_changed();
      } catch (const exception& e) {
        print_and_clear_exception("Value: squelching exception in an Action",e);
      }
    }
  }
}

void ValueBase::signal() const {
  // Collect all actions which depend on us, destroying the associated links in the process.
  Link* link = actions;
  while (link) {
    Link* next = link->value_next;
    if (!link->action->executing) { // If the action is already executing, don't retrigger it
      // Remove from our linked list
      *link->value_prev = link->value_next;
      if (link->value_next)
        link->value_next->value_prev = link->value_prev;
      // Add it to the pending list
      link->value_next = pending;
      link->value_prev = &pending;
      if (pending)
        pending->value_prev = &link->value_next;
      pending = link;
    }
    link = next;
  }

  // Send signals
  signal_pending();
}

void ValueBase::pull() const {
  // If there are any pending signals, send them
  signal_pending();

  // If a node is currently being evaluated, it now depends on us
  if (Action::current)
    Action::current->depend_on(*this);

  // Update if necessary
  if (dirty_) {
    try {
      update();
    } catch (exception& e) {
      dirty_ = false;
      error = ExceptionValue(e);
      throw;
    }
    GEODE_ASSERT(!dirty_);
  } else if (error)
    error.throw_();
}

vector<Ref<const ValueBase>> ValueBase::dependents() const {
  vector<Ref<const ValueBase>> result;
  for (ValueBase::Link* link=actions; link; link=link->value_next) {
    // make sure we have a value here, don't put other things in the
    auto value_p = dynamic_cast<ValueBase*>(link->action);
    if (value_p)
      result.push_back(ref(*value_p));
  }
  return result;
}

vector<Ref<const ValueBase>> ValueBase::all_dependents() const {
  auto depv = dependents();
  unordered_set<Ref<const ValueBase>, Hasher> deps(depv.begin(), depv.end());
  unordered_set<Ref<const ValueBase>, Hasher> incoming = deps;
  while (!incoming.empty()) {
    auto depdeps = (*incoming.begin())->dependents();
    incoming.erase(incoming.begin());
    for (auto dep : depdeps) {
      if (!deps.count(dep)) {
        incoming.insert(dep);
        deps.insert(dep);
      }
    }
  }
  return vector<Ref<const ValueBase>>(deps.begin(), deps.end());
}

vector<Ref<const ValueBase>> ValueBase::all_dependencies() const {
  auto depv = dependencies();
  unordered_set<Ref<const ValueBase>, Hasher> deps(depv.begin(), depv.end());
  unordered_set<Ref<const ValueBase>, Hasher> incoming = deps;
  while (!incoming.empty()) {
    auto depdeps = (*incoming.begin())->dependencies();
    incoming.erase(incoming.begin());
    for (auto dep : depdeps) {
      if (!deps.count(dep)) {
        incoming.insert(dep);
        deps.insert(dep);
      }
    }
  }
  return vector<Ref<const ValueBase>>(deps.begin(), deps.end());
}


#ifdef GEODE_PYTHON
// For testing purposes
static ValueRef<int> value_test(ValueRef<int> value) {
  return value;
}
static void value_ptr_test(Ptr<Value<int>> value) {}
#endif

bool ValueBase::is_prop() const {
  return dynamic_cast<const PropBase*>(this)!=0;
}

// for backwards compatibility with previously un-named values
const string& ValueBase::name() const { return name_; }

// The following exist only for python purposes: they throw exceptions if the ValueBase isn't a PropBase.
PropBase& ValueBase::prop() {
  if (PropBase* prop = dynamic_cast<PropBase*>(this))
    return *prop;
  throw TypeError("Prop expected, got Value");
}
const PropBase& ValueBase::prop() const {
  return const_cast_(*this).prop();
}
ValueBase& ValueBase::set_help(const string& h)     { prop().help = h; return *this; }
ValueBase& ValueBase::set_category(const string& c) { prop().category = c; return *this; }
ValueBase& ValueBase::set_hidden(bool h)            { prop().hidden = h; return *this; }
ValueBase& ValueBase::set_required(bool r)          { prop().required = r; return *this; }
ValueBase& ValueBase::set_abbrev(char a)            { prop().abbrev = a; return *this; }
const string& ValueBase::get_help() const     { return prop().help; }
const string& ValueBase::get_category() const { return prop().category; }
bool ValueBase::get_hidden() const            { return prop().hidden; }
bool ValueBase::get_required() const          { return prop().required; }
char ValueBase::get_abbrev() const            { return prop().abbrev; }

#ifdef GEODE_PYTHON
ValueBase& ValueBase::set_python(PyObject* value_)  { prop().set_python(value_); return *this; }
ValueBase& ValueBase::set_allowed(PyObject* v)      { prop().set_allowed_python(v); return *this; }
ValueBase& ValueBase::set_min_py(PyObject* m)       { prop().set_min_python(m); return *this; }
ValueBase& ValueBase::set_max_py(PyObject* m)       { prop().set_max_python(m); return *this; }
ValueBase& ValueBase::set_step_py(PyObject* s)      { prop().set_step_python(s); return *this; }
Ref<> ValueBase::get_default() const      { return prop().default_python(); }
Ref<> ValueBase::get_min_py() const          { return prop().get_min_python(); }
Ref<> ValueBase::get_max_py() const          { return prop().get_max_python(); }
Ref<> ValueBase::get_allowed() const      { return prop().allowed_python(); }
#endif

}
using namespace geode;

void wrap_value_base() {
#ifdef GEODE_PYTHON
  typedef ValueBase Self;
  Class<Self>("Value")
    .GEODE_CALL()
    .GEODE_GET(name)
    .GEODE_METHOD(dirty)
    .GEODE_METHOD(dump)
    .GEODE_METHOD(dependents)
    .GEODE_METHOD(all_dependents)
    .GEODE_METHOD(dependencies)
    .GEODE_METHOD(all_dependencies)
    .GEODE_METHOD(signal)
    .GEODE_METHOD(is_prop)
    // The following work only if the object is a Prop
    .GEODE_METHOD_2("set",set_python)
    .property("help",&Self::get_help)
    .property("hidden",&Self::get_hidden)
    .property("required",&Self::get_required)
    .property("category",&Self::get_category)
    .property("abbrev",&Self::get_abbrev)
    .property("allowed",&Self::get_allowed)
    .property("default",&Self::get_default)
    .GEODE_METHOD(set_help)
    .GEODE_METHOD(set_hidden)
    .GEODE_METHOD(set_required)
    .GEODE_METHOD(set_abbrev)
    .GEODE_METHOD(set_allowed)
    .GEODE_METHOD(set_category)
    .GEODE_METHOD_2("set_min",set_min_py)
    .GEODE_METHOD_2("set_max",set_max_py)
    .GEODE_METHOD_2("get_min",get_min_py)
    .GEODE_METHOD_2("get_max",get_max_py)
    .GEODE_METHOD_2("set_step",set_step_py)
    ;

  GEODE_FUNCTION(value_test)
  GEODE_FUNCTION(value_ptr_test)
#endif
}
