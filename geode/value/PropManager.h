//#####################################################################
// Class PropManager
//#####################################################################
#pragma once

#include <geode/utility/format.h>
#include <geode/utility/unordered.h>
#include <geode/value/Prop.h>
#include <stdexcept>
#include <string>
#include <vector>
namespace geode {

class PropManager : public Object {
public:
  GEODE_NEW_FRIEND

  typedef unordered_map<string,Ref<PropBase>> Props;
  Props items;
  vector<string> order;
  bool frozen; // Set to true to make temporarily immutable

protected:
  GEODE_CORE_EXPORT PropManager();
public:
  ~PropManager();

  // Add an existing property if the name or new, or return the existing one.
  GEODE_CORE_EXPORT PropBase& add(PropBase& prop);

  template<class T> Prop<T>& add(const PropRef<T>& prop) {
    return static_cast<Prop<T>&>(add(prop.self));
  }

  // Add a new property with the given type and default
  template<class T> Prop<T>& add(const string& name, const T& default_) {
    return add(PropRef<T>(name,default_));
  }

  // Check if a property exists
  GEODE_CORE_EXPORT bool contains(const string& name) const;

  // Get a property by name, throwing an exception if none is found or the type doesn't match
  GEODE_CORE_EXPORT PropBase& get(const string& name) const;
  GEODE_CORE_EXPORT PropBase& get(const string& name, const type_info& type) const;

  // Get a property by name, return 0 if none is found.  If type is specified, throw an exception if the type doesn't match.
  GEODE_CORE_EXPORT PropBase* get_ptr(const string& name) const;
  GEODE_CORE_EXPORT PropBase* get_ptr(const string& name, const type_info& type) const;

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
  GEODE_CORE_EXPORT Prop<string>& add(const string& name, const char* default_);
  GEODE_CORE_EXPORT Prop<string>& get_or_add(const string& name, const char* default_);

#if 0 // Value python support
  PropBase& add_python(const string& name, PyObject* default_);
  PropBase& get_or_add_python(const string& name, PyObject* default_);
  PropBase& getattr(const string& name) const;
#endif
};

}
