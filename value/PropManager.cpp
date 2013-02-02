#include <other/core/value/PropManager.h>
#include <other/core/python/Class.h>
#include <other/core/python/stl.h>
namespace other {

using std::pair;
using std::cout;
using std::endl;

OTHER_DEFINE_TYPE(PropManager)

PropManager::PropManager()
  : frozen(false) {}

PropManager::~PropManager() {}

PropBase& PropManager::add(PropBase& prop) {
  const string& name = prop.name_();
  if (name.find('-')!=string::npos)
    throw ValueError(format("prop name '%s' contains a dash; use an underscore instead",name));
  Props::iterator it = items.find(name);
  if (it == items.end()) {
    if (frozen)
      throw TypeError(format("prop manager is frozen, can't create new property '%s'",name));
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
  return it->second;
}

bool PropManager::contains(const string& name) const {
  return items.count(name)!=0;
}

PropBase* PropManager::get_ptr(const string& name) const {
  const auto it = items.find(name);
  if (it==items.end())
    return 0;
  return &*it->second;
}

PropBase& PropManager::get(const string& name) const {
  if (auto value = get_ptr(name))
    return *value;
  throw KeyError(format("No value named '%s' exists", name));
}

PropBase& PropManager::get(const string& name, const type_info& type) const {
  if (auto value = get_ptr(name,type))
    return *value;
  throw KeyError(format("No value named '%s' exists", name));
}

PropBase* PropManager::get_ptr(const string& name, const type_info& type) const {
  if (auto prop = get_ptr(name)) {
    if (prop->base().is_type(type))
      return prop;
    throw TypeError(format("Property '%s' accessed with type %s, but has type %s", name,type.name(),prop->type().name()));
  }
  return 0;
}

Prop<string>& PropManager::add(const string& name, const char* default_) {
  return add(name,string(default_));
}

Prop<string>& PropManager::get_or_add(const string& name, const char* default_) {
  return get_or_add(name,string(default_));
}

#ifdef OTHER_PYTHON
PropBase& PropManager::add_python(const string& name, PyObject* default_) {
  return add(make_prop(name,default_));
}

PropBase& PropManager::get_or_add_python(const string& name, PyObject* default_) {
  if (auto value = get_ptr(name))
    return *value;
  return add_python(name,default_);
}
#endif

}
using namespace other;

void wrap_prop_manager() {
#ifdef OTHER_PYTHON
  typedef PropManager Self;
  Class<Self>("PropManager")
    .OTHER_INIT()
    .method("add_existing",static_cast<PropBase&(Self::*)(PropBase&)>(&Self::add))
    .method("get",static_cast<PropBase&(Self::*)(const string&)const>(&Self::get))
    .OTHER_METHOD_2("add",add_python)
    .OTHER_METHOD_2("get_or_add",get_or_add_python)
    .OTHER_METHOD(contains)
    .OTHER_CONST_FIELD(items)
    .OTHER_CONST_FIELD(order)
    .OTHER_FIELD(frozen)
    ;
#endif
}
