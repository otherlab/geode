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

PropManager::PropManager() {}

PropManager::~PropManager() {}

PropBase& PropManager::add(PropBase& prop) {
  const string& name = prop.name_();
  if (name.find('-')!=string::npos)
    throw ValueError(format("prop name '%s' contains a dash; use an underscore instead",name));
  Props::iterator it = items.find(name);
  if (it == items.end()) {
    items.insert(make_pair(name,ref(prop)));
    order.push_back(name);
    return prop;
  }
  const type_info &old_type = it->second->type(),
                  &new_type = prop.type();
  if (new_type!=old_type && strcmp(new_type.name(),old_type.name()) != 0)
    throw TypeError(format("Property '%s' accessed with type %s, but has type %s",name,new_type.name(),old_type.name()));
  if (!prop.same_default(*it->second))
    throw ValueError(format("Trying to add property '%s' with default %s, but old default was %s",name,prop.value_str(true),it->second->value_str(true)));
  return *it->second;
}

Prop<string>& PropManager::add(string const &name, const char* default_,  bool required, bool hidden) {
  return add(name,string(default_),required,hidden);
}

Prop<string>& PropManager::get_or_add(string const &name, const char* default_) {
  return get_or_add(name,string(default_));
}

PropBase& PropManager::add_python(string const &name, PyObject* default_) {
  return add(make_prop(name,default_));
}

PropBase& PropManager::get(string const &name) const {
  auto it = items.find(name);
  if (it == items.end())
    throw KeyError(format("No property named '%s' exists", name));
  return it->second;
}

PropBase& PropManager::get_or_add_python(string const &name, PyObject* default_) {
  auto it = items.find(name);
  if (it != items.end()) return it->second;
  else return add_python(name, default_);
}

}

void wrap_prop_manager() {
  using namespace other;
  typedef PropManager Self;
  Class<Self>("PropManager")
    .OTHER_INIT()
    .OTHER_METHOD_2("add",add_python)
    .method("add_existing",static_cast<PropBase&(Self::*)(PropBase&)>(&Self::add))
    .method("get",static_cast<PropBase&(Self::*)(const string&)const>(&Self::get))
    .OTHER_METHOD_2("get_or_add",get_or_add_python)
    .OTHER_CONST_FIELD(items)
    .OTHER_CONST_FIELD(order)
    ;
}
