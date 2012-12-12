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

class PropManager: public Object {
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)

  typedef unordered_map<string,Ref<PropBase>> Props;
  Props items;
  vector<string> order;

private:
  OTHER_CORE_EXPORT PropManager() ;
public:
  ~PropManager();

  OTHER_CORE_EXPORT PropBase& add(PropBase& prop) ;

  template<class T> Prop<T>& add(PropRef<T> prop) {
    return static_cast<Prop<T>&>(add(prop.self));
  }

  // Add an uninitialized property (default constructor is used), which will be flagged as
  // 'required' when passed to the parser.
  template<class T> Prop<T>& add(string const &name) {
    return add(name, T(), true, false);
  }

  template<class T> Prop<T>& add(string const &name, T const &default_,  bool required=false, bool hidden=false) {
    PropRef<T> prop(name,default_);
    prop->hidden = hidden;
    prop->required = required;
    return add(prop);
  }

  // Turn char* to string
  OTHER_CORE_EXPORT Prop<string>& add(string const &name, const char* default_,  bool required=false, bool hidden=false) ;

  // Gets a property by name, throwing an exception if none found
  OTHER_CORE_EXPORT PropBase& get(string const &name) const ;

  // Gets a property by name, throwing an exception if none found (or not the right type)
  template<class T> Prop<T>& get(string const &name) const {
    Props::const_iterator it = items.find(name);
    if (it == items.end()) {
      throw KeyError(format("No property named '%s' exists", name));
    } else {
      if(Prop<T>* p = it->second->cast<T>())
        return *p;
      else
        throw TypeError(format("Property '%s' accessed with type %s, but has type %s", name,typeid(T).name(),it->second->type().name()));
    }
  }

  template<class T> Prop<T>& get_or_add(string const &name, T const &default_) {
    Props::const_iterator it = items.find(name);
    if (it == items.end()) {
      return add(name, default_);
    } else {
      if(Prop<T>* p = it->second->cast<T>())
        return *p;
      else
        throw TypeError(format("Property '%s' accessed with type %s, but has type %s", name,typeid(T).name(),it->second->type().name()));
    }
  }

  // Turn char* to string
  OTHER_CORE_EXPORT Prop<string>& get_or_add(string const &name, const char* default_) ;

#ifdef OTHER_PYTHON
  PropBase& add_python(string const &name, PyObject* default_);
  PropBase& get_or_add_python(string const &name, PyObject* default_);
#endif
};

}
