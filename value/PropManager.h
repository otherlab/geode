//#####################################################################
// Class PropManager
//#####################################################################
#pragma once

#include <other/core/utility/format.h>
#include <other/core/utility/tr1.h>
#include <other/core/value/Prop.h>
#include <stdexcept>
#include <string>
#include <vector>
namespace other {

class PropManager : public Object {
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)

  typedef unordered_map<string,Ref<PropBase>> Props;
  Props items;
  vector<string> order;

protected:
  OTHER_CORE_EXPORT PropManager();
public:
  ~PropManager();

  // Add an existing property if the name or new, or return the existing one.
  OTHER_CORE_EXPORT PropBase& add(PropBase& prop);

  template<class T> Prop<T>& add(const PropRef<T>& prop) {
    return static_cast<Prop<T>&>(add(prop.self));
  }

  // Add a new property with the given type and default
  template<class T> Prop<T>& add(const string& name, const T& default_) {
    return add(PropRef<T>(name,default_));
  }

  // Get a property by name, throwing an exception if none is found or the type doesn't match
  OTHER_CORE_EXPORT PropBase& get(const string& name) const;
  OTHER_CORE_EXPORT PropBase& get(const string& name, const type_info& type) const;

  // Get a property by name, return 0 if none is found.  If type is specified, throw an exception if the type doesn't match.
  OTHER_CORE_EXPORT PropBase* get_ptr(const string& name) const;
  OTHER_CORE_EXPORT PropBase* get_ptr(const string& name, const type_info& type) const;

  // Get a property by name, throwing an exception if none is found (or not the right type)
  template<class T> Prop<T>& get(const string& name) const {
    return static_cast<Prop<T>&>(get(name,typeid(T)));
  }

  template<class T> Prop<T>& get_or_add(const string& name, const T& default_) {
    if (auto prop = get_ptr(name,typeid(T)))
      return static_cast<Prop<T>&>(*prop);
    return add(name,default_);
  }

  // Turn char* to string
  OTHER_CORE_EXPORT Prop<string>& add(const string& name, const char* default_);
  OTHER_CORE_EXPORT Prop<string>& get_or_add(const string& name, const char* default_);

#ifdef OTHER_PYTHON
  PropBase& add_python(const string& name, PyObject* default_);
  PropBase& get_or_add_python(const string& name, PyObject* default_);
#endif
};

}
