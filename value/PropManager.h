//#####################################################################
// Class PropManager
//#####################################################################
#pragma once

#include <other/core/utility/format.h>
#include <other/core/value/Prop.h>
#include <tr1/unordered_map>
#include <stdexcept>
#include <string>
#include <vector>
namespace other {

using std::tr1::unordered_map;

class OTHER_EXPORT PropManager: public Object {
public:
  OTHER_DECLARE_TYPE

  typedef unordered_map<string,Ref<PropBase> > Props;
  Props props_;
  vector<string> order_;

private:
  PropManager() {}
  static PropManager& instance();
public:

  static Props& props() {
    return instance().props_;
  }

  static vector<string>& order() {
    return instance().order_;
  }

  static PropBase& add(Ref<PropBase> prop);

  template<class T> static Prop<T>& add(PropRef<T> prop) {
    return static_cast<Prop<T>&>(add(prop.self));
  }

  // adds an uninitialized property (default constructor is used), which will be flagged as
  // 'required' when passed to the parser.
  template<class T> static Prop<T>& add(string const &name) {
    return add(name, T(), true, false);
  }

  template<class T> static Prop<T>& add(string const &name, T const &default_,  bool required=false, bool hidden=false) {
    PropRef<T> prop(name,default_);
    prop->hidden = hidden;
    prop->required = required;
    return add(prop);
  }

  // gets a property by name, throws an exception if none found (or not the right type)
  template<class T> static Prop<T>& get(string const &name) {
    const Props& props = instance().props_;
    Props::const_iterator it = props.find(name);
    if (it == props.end()) {
      throw KeyError(format("No property named '%s' exists", name));
    } else {
      if(Prop<T>* p = it->second->cast<T>())
        return *p;
      else
        throw TypeError(format("Property '%s' accessed with type %s, but has type %s", name,typeid(T).name(),it->second->type().name()));
    }
  }

  template<class T> static Prop<T>& get_or_add(string const &name, T const &default_) {
    const Props& props = instance().props_;
    Props::const_iterator it = props.find(name);
    if (it == props.end()) {
      return add(name, default_);
    } else {
      if(Prop<T>* p = it->second->cast<T>())
        return *p;
      else
        throw TypeError(format("Property '%s' accessed with type %s, but has type %s", name,typeid(T).name(),it->second->type().name()));
    }
  }

  static PropBase& add_python(string const &name, PyObject* default_);
  static PropBase& get_python(string const &name);
  static PropBase& get_or_add_python(string const &name, PyObject* default_);

  // returns a copy of props for python
  static PyObject* get_python_items();
};

}
