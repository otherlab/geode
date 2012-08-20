#include <other/core/array/Array.h>
#include <other/core/value/Value.h>
#include <other/core/value/Action.h>
#include <other/core/value/Prop.h>
#include <other/core/value/convert.h>
#include <other/core/python/Class.h>
#include <other/core/python/module.h>
#include <other/core/python/wrap_call.h>
#include <other/core/utility/const_cast.h>
#include <iostream>

#include <other/core/python/stl.h>

namespace other{

using std::cout;
using std::cerr;
using std::endl;
using std::exception;

OTHER_DEFINE_TYPE(ValueBase)

// Link list of actions which have pending signals.  The fact that this is a two-sided doubly-linked list
// is important, since Actions need to be able to delete their links from the pending list if they
// destruct during dependency propagation.
ValueBase::Link* ValueBase::pending = 0;

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
      } catch(const exception& e) {
        cerr << "ValueBase::pull: squelching an exception thrown by an action: " << e.what() << endl;
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
    const_cast_(Action::current)->depend_on(*this);

  // Update if necessary
  if (dirty_) {
    try {
      update();
    } catch (exception& e) {
      dirty_ = false;
      error = ExceptionValue(e);
      throw;
    }
    OTHER_ASSERT(!dirty_);
  } else if (error)
    error.throw_();
}

Ref<> ValueBase::dump_dependencies() const {
  return to_python_ref(get_dependencies());
}

// For testing purposes
ValueRef<int> value_test(ValueRef<int> value) {
  return value;
}

bool ValueBase::is_prop() const {
  return dynamic_cast<const PropBase*>(this)!=0;
}

// The following exist only for python purposes: they throw exceptions if the ValueBase isn't a PropBase.
PropBase& ValueBase::prop() {
  if (PropBase* prop = dynamic_cast<PropBase*>(this))
    return *prop;
  throw TypeError("Prop expected, got Value");
}
const PropBase& ValueBase::prop() const {
  return const_cast_(*this).prop();
}
ValueBase& ValueBase::set(PyObject* value_)         { prop().set(value_); return *this; }
ValueBase& ValueBase::set_help(const string& h)     { prop().help = h; return *this; }
ValueBase& ValueBase::set_category(const string& c) { prop().category = c; return *this; }
ValueBase& ValueBase::set_hidden(bool h)            { prop().hidden = h; return *this; }
ValueBase& ValueBase::set_required(bool r)          { prop().required = r; return *this; }
ValueBase& ValueBase::set_abbrev(char a)            { prop().abbrev = a; return *this; }
ValueBase& ValueBase::set_allowed(PyObject* v)      { prop().set_allowed_python(v); return *this; }
ValueBase& ValueBase::set_min_py(PyObject* m)       { prop().set_min_python(m); return *this; }
ValueBase& ValueBase::set_max_py(PyObject* m)       { prop().set_max_python(m); return *this; }
ValueBase& ValueBase::set_step_py(PyObject* s)      { prop().set_step_python(s); return *this; }
const string& ValueBase::get_name() const     { return prop().name; }
const string& ValueBase::get_help() const     { return prop().help; }
const string& ValueBase::get_category() const { return prop().category; }
bool ValueBase::get_hidden() const            { return prop().hidden; }
bool ValueBase::get_required() const          { return prop().required; }
char ValueBase::get_abbrev() const            { return prop().abbrev; }
PyObject* ValueBase::get_default() const      { return prop().default_python(); }
PyObject* ValueBase::get_allowed() const      { return prop().allowed_python(); }

// Instantiate common versions
template class Value<bool>;
template class Value<int>;
template class Value<double>;
template class Value<string>;
template class Value<Vector<real,2>>;

template class Value<Vector<real,3>>;
template class Value<Vector<real,4>>;

}
using namespace other;

void wrap_value_base() {
  typedef ValueBase Self;
  Class<Self>("Value")
    .OTHER_CALL(PyObject*)
    .OTHER_METHOD(dirty)
    .OTHER_METHOD(dump)
    .OTHER_METHOD(get_dependencies)
    .OTHER_METHOD(signal)
    .OTHER_METHOD(is_prop)
    // The following work only if the object is a Prop
    .OTHER_METHOD(set)
    .property("name",&Self::get_name)
    .property("help",&Self::get_help)
    .property("hidden",&Self::get_hidden)
    .property("required",&Self::get_required)
    .property("category",&Self::get_category)
    .property("abbrev",&Self::get_abbrev)
    .property("allowed",&Self::get_allowed)
    .property("default",&Self::get_default)
    .OTHER_METHOD(set_help)
    .OTHER_METHOD(set_hidden)
    .OTHER_METHOD(set_required)
    .OTHER_METHOD(set_abbrev)
    .OTHER_METHOD(set_allowed)
    .OTHER_METHOD(set_category)
    .OTHER_METHOD_2("set_min",set_min_py)
    .OTHER_METHOD_2("set_max",set_max_py)
    .OTHER_METHOD_2("set_step",set_step_py)
    ;

  OTHER_FUNCTION(value_test)
}
