//#####################################################################
// Class Prop
//#####################################################################
#pragma once

#include <other/core/value/forward.h>
#include <other/core/value/Value.h>
#include <other/core/math/clamp.h>
#include <other/core/python/Ref.h>
#include <other/core/python/Ptr.h>
#include <other/core/python/try_convert.h>
#include <other/core/python/stl.h>
#include <other/core/structure/forward.h>
#include <other/core/utility/CopyConst.h>
#include <other/core/utility/format.h>
#include <other/core/utility/const_cast.h>
#include <other/core/utility/stl.h>
#include <other/core/utility/str.h>
#include <other/core/vector/Vector.h>
#include <boost/type_traits/remove_const.hpp>
#include <boost/scoped_ptr.hpp>
namespace other {

using std::string;
using std::type_info;
using std::vector;
using std::ostringstream;
using boost::scoped_ptr;

class PropManager;

class OTHER_CORE_CLASS_EXPORT PropBase { // Need OTHER_CORE_EXPORT for typeid
protected:
  OTHER_CORE_EXPORT PropBase();
private:
  PropBase(const PropBase&); // noncopyable
  void operator=(const PropBase&);
public:
  OTHER_CORE_EXPORT virtual ~PropBase();

  virtual const ValueBase& base() const = 0;
  virtual bool same_default(PropBase& other) const = 0;
  virtual string value_str(bool use_default = false) const = 0;

#ifdef OTHER_PYTHON
  virtual void set(PyObject* value_) = 0;
  virtual PyObject* default_python() const = 0;
  virtual void set_allowed_python(PyObject* values) = 0;
  virtual PyObject* allowed_python() const = 0;
  virtual void set_min_python(PyObject* v) = 0;
  virtual void set_max_python(PyObject* v) = 0;
  virtual void set_step_python(PyObject* v) = 0;
#endif

  const type_info& type() const {
    return base().type();
  }

  const string& name_() const { // Can't be named name due to ambiguity in gcc 4.6
    return base().name;
  }

  template<class T> Prop<T>* cast() {
    const type_info &goal = typeid(Prop<T>),
                    &self = typeid(*this);
    if (goal==self || !strcmp(goal.name(),self.name())) // Use string comparison to avoid symbol visibility issues
      return static_cast<Prop<T>*>(this);
    return 0;
  }

  string help;
  bool hidden;
  bool required;
  char abbrev;
  string category; //TODO: nested categorization? include anything dependency-graph based?

  OTHER_CORE_EXPORT void dump(int indent) const ;
};

inline Ref<PropBase> ref(PropBase& prop) {
  return Ref<PropBase>(prop,to_python(prop.base()));
}

template<class T,bool enable> struct PropClamp;

template<class T> struct PropClamp<T,false> {
  Prop<T>& self() {
    return static_cast<Prop<T>&>(*this);
  }
  const T& clamp(const T& value) {
    return value;
  }
  Prop<T>& copy_range_from(const PropClamp& p) {
    return self();
  }

  void set_min_python(PyObject* v){throw ValueError(format("non-clampable prop cannot set min"));}
  void set_max_python(PyObject* v){ throw ValueError(format("non-clampable prop cannot set max"));}
  void set_step_python(PyObject* v){ throw ValueError(format("non-clampable prop cannot set step"));}
};

template<class T> struct PropClamp<T,true> {
  T min,max,step;
private:
  typedef PropClamp Self;
  scoped_ptr<Tuple<PropRef<T>,Ref<Listen>,real>> prop_min, prop_max, prop_step;

protected:
  OTHER_CORE_EXPORT PropClamp();
public:
  OTHER_CORE_EXPORT ~PropClamp();

  Prop<T>& self() {
    return static_cast<Prop<T>&>(*this);
  }

  T clamp(T value) const {
    return other::clamp(value,min,max);
  }

  Prop<T>& set_min(const T& m){
    min = m;
    return self();
  }

  Prop<T>& set_max(const T& m) {
    max = m;
    return self();
  }

  Prop<T>& set_step(const T& s){
    step = s;
    return self();
  }

#ifdef OTHER_PYTHON
  void set_min_python(PyObject* v){
    set_min(from_python<T>(v));
  }

  void set_max_python(PyObject* v){
    set_max(from_python<T>(v));
  }

  void set_step_python(PyObject* v){
    set_step(from_python<T>(v));
  }
#endif

  OTHER_CORE_EXPORT Prop<T>& set_min(const PropRef<T> p, real alpha = 1);
  OTHER_CORE_EXPORT Prop<T>& set_max(const PropRef<T> p, real alpha = 1);

  Prop<T>& copy_range_from(const PropClamp& p) {
    set_min(p.min);
    set_max(p.max);
    return self();
  }

private:
  void minimize();
  void maximize();
};

template<class T> class OTHER_CORE_CLASS_EXPORT Prop : public Value<T>, public PropBase, public PropClamp<T,has_clamp<T>::value>
{
public:
  OTHER_NEW_FRIEND
  typedef Value<T> Base;
  friend class PropManager;
  friend struct PropClamp<T,has_clamp<T>::value>;
  typedef PropClamp<T,has_clamp<T>::value> Clamp;
  using Base::name;

protected:
  Prop(string const& name, const T& value_) 
    : PropBase(), default_(value_)
  {
    this->set_name(name);
    this->set_value(value_);
  }

  // Properties never go invalid, so update should never be called
  void update() const {
    OTHER_FATAL_ERROR();
  }
public:

  const T default_;
  vector<T> allowed;

  void set(const T& value_) {
    if (!Equals<T>::eval(*Base::value,value_)) {
      if(allowed.size() && !other::contains(allowed,value_))
        throw ValueError("value not in allowed values for " + name);
      this->set_value(Clamp::clamp(value_));
    }
  }

  Prop<T>& set_help(const string& h){
    help = h;
    return *this;
  }

  Prop<T>& set_category(const string& c){
    category = c;
    return *this;
  }

  Prop<T>& set_hidden(bool h){
    hidden = h;
    return *this;
  }

  Prop<T>& set_required(bool r){
    required = r;
    return *this;
  }

  Prop<T>& set_abbrev(char a){
    abbrev = a;
    return *this;
  }

  Prop<T>& set_allowed(const vector<T>& v){
    allowed = v;
    return *this;
  }

#ifdef OTHER_PYTHON

  void set(PyObject* value_) {
    set(try_from_python<T>(value_));
  }

  void set_allowed_python(PyObject* values){
    set_allowed(try_from_python<vector<T>>(values));
  }

  PyObject* allowed_python() const {
    return try_to_python(allowed);
  }

  void set_min_python(PyObject* m){
    Clamp::set_min_python(m);
  }

  void set_max_python(PyObject* m){
    Clamp::set_max_python(m);
  }

  void set_step_python(PyObject* s){
    Clamp::set_step_python(s);
  }

  PyObject* default_python() const {
    return try_to_python(default_);
  }

#endif

  const ValueBase& base() const {
    return *this;
  }

  // Look at a property without adding a dependency graph node
  const T& peek() const {
    return *this->value;
  }

  // WARNING: This will give a mutable reference to the contained prop; to keep sanity in the
  // dependency graph, you will have to call signal.  If you change the mutable reference
  // and an exception prevents you from calling signal, an obscure bug will result.  Use at
  // your own risk.
  T& mutate() {
    return *this->value;
  }

  bool same_default(PropBase& other_) const {
    Prop* other = other_.cast<T>();
    return other && Equals<T>::eval(default_,other->default_);
  }

  string value_str(bool use_default) const {
    return str(use_default?default_:*this->value);
  }

  void dump(int indent) const {
    PropBase::dump(indent);
  }

  vector<Ref<const ValueBase>> dependencies() const {
    return vector<Ref<const ValueBase>>();
  }
};

template<class T> class PropRef {
public:
  typedef typename boost::remove_const<T>::type type;
  typedef typename CopyConst<Prop<type>,T>::type prop_type;
  Ref<typename CopyConst<Prop<type>,T>::type> self;

  PropRef(const string& name, const T& value)
    : self(new_<Prop<T>>(name,value)) {}

  PropRef(typename CopyConst<Prop<type>,T>::type& self)
    : self(other::ref(self)) {}

  prop_type* operator->() const {
    return &*self;
  }

  prop_type& operator*() const {
    return *self;
  }

  operator prop_type&() const {
    return *self;
  }

  const T& operator()() const {
    return (*self)();
  }

  bool operator==(const PropRef<type>& p) const{
    return self == p.self;
  }

  PropRef<T> clone_prop() const {
    PropRef<T> result(self->name,self->default_);
    result->set_allowed(self->allowed);
    result->copy_range_from(self);
    result->set((*this)());
    return result;
  }
};

template<class T> inline std::ostream& operator<<(std::ostream& output, const PropRef<T>& ref) {
  return output<<ref();
}

#ifdef OTHER_PYTHON

OTHER_CORE_EXPORT PyObject* to_python(const PropBase& prop);
OTHER_CORE_EXPORT PyObject* ptr_to_python(const PropBase* prop);
OTHER_CORE_EXPORT PropBase& prop_from_python(PyObject* object, const type_info& type);
OTHER_CORE_EXPORT Ref<PropBase> make_prop(string const&, PyObject* value);

template<class T> PyObject* ptr_to_python(const Prop<T>* prop) {
  return ptr_to_python(static_cast<const PropBase*>(prop));
}

template<class T> PyObject* to_python(const PropRef<T>& prop) {
  return to_python(static_cast<PropBase&>(prop.self));
}

template<> struct FromPython<PropBase&> { static PropBase& convert(PyObject* object); };

template<class T> struct FromPython<PropRef<T>> {
  static PropRef<T> convert(PyObject* object) {
    return static_cast<Prop<T>&>(prop_from_python(object,typeid(T)));
  }
};

#endif

}
