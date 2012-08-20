#include <other/core/value/PropManager.h>
#include <other/core/python/Ref.h>
#include <other/core/python/Class.h>
#include <other/core/python/repr.h>
#include <other/core/python/stl.h>
namespace other {

using std::pair;
using std::cout;
using std::endl;

  OTHER_DEFINE_TYPE(PropManager)

  static Ref<PropManager> pm_instance = new_<PropManager>();

  PropManager &PropManager::instance() {
    return *pm_instance;
  }

  Ref<PropManager> get_prop_manager() {
    return pm_instance;
  }

  PropBase& PropManager::add(Ref<PropBase> prop) {
    if (prop->name.find('-')!=string::npos)
      throw ValueError(format("prop name '%s' contains a dash; use an underscore instead",prop->name));
    Props& props = instance().props_;
    Props::iterator it = props.find(prop->name);
    if (it == props.end()) {
      props.insert(make_pair(prop->name,prop));
      order().push_back(prop->name);
      return *prop;
    }
    const type_info &old_type = it->second->type(),
                    &new_type = prop->type();
    if (new_type!=old_type && strcmp(new_type.name(),old_type.name()) != 0)
      throw TypeError(format("Property '%s' accessed with type %s, but has type %s",prop->name,new_type.name(),old_type.name()));
    if (!prop->same_default(*it->second))
      throw ValueError(format("Trying to add property '%s' with default %s, but old default was %s",prop->name,prop->value_str(true),it->second->value_str(true)));
    return *it->second;
  }

  PropBase& PropManager::add_python(string const &name, PyObject* default_) {
    return add(make_prop(name,default_));
  }

  PropBase& PropManager::get_python(string const &name) {
    Props& props = instance().props_;
    Props::iterator it = props.find(name);
    if (it == props.end())
      throw KeyError(format("No property named '%s' exists", name));
    return it->second;
  }

  PropBase& PropManager::get_or_add_python(string const &name, PyObject* default_) {
    Props& props = instance().props_;
    Props::iterator it = props.find(name);
    if (it != props.end()) return it->second;
    else return add_python(name, default_);
  }

  PyObject* PropManager::get_python_items() {
    Props& props = instance().props_;

    // make a python dict, and copy props
    PyObject *dict = PyDict_New();

    typedef pair<const string, Ref<PropBase> > nptype;
    for (nptype &np : props) {
      PyDict_SetItemString(dict, np.first.c_str(), to_python(*np.second));
    }

    return dict;
  }
}

void wrap_prop_manager() {
  using namespace other;
  typedef PropManager Self;
  Class<Self>("PropManager")
    .OTHER_METHOD_2("add",add_python)
    .method("add_existing",static_cast<PropBase&(*)(Ref<PropBase>)>(&Self::add))
    .OTHER_METHOD_2("get",get_python)
    .OTHER_METHOD_2("get_or_add",get_or_add_python)
    .OTHER_METHOD_2("items",get_python_items)
    .OTHER_METHOD(order)
    ;
}
