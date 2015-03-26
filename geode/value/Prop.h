//#####################################################################
// Class Prop
//#####################################################################
#pragma once

#include <geode/value/forward.h>
#include <geode/value/Value.h>
#include <geode/math/clamp.h>
#include <geode/python/Ref.h>
#include <geode/python/Ptr.h>
#include <geode/python/try_convert.h>
#include <geode/python/stl.h>
#include <geode/structure/forward.h>
#include <geode/utility/CopyConst.h>
#include <geode/utility/format.h>
#include <geode/utility/const_cast.h>
#include <geode/utility/stl.h>
#include <geode/utility/str.h>
#include <geode/utility/type_traits.h>
#include <geode/vector/Vector.h>
namespace geode {

using std::string;
using std::type_info;
using std::vector;
using std::ostringstream;
class PropManager;

class GEODE_CORE_CLASS_EXPORT PropBase { // Need GEODE_CORE_EXPORT for typeid
protected:
  GEODE_CORE_EXPORT PropBase();
private:
  PropBase(const PropBase&); // noncopyable
  void operator=(const PropBase&);
public:
  GEODE_CORE_EXPORT virtual ~PropBase();

  virtual const ValueBase& base() const = 0;
  virtual bool same_default(PropBase& other) const = 0;
  virtual string value_str(bool use_default = false) const = 0;

#ifdef GEODE_PYTHON
  virtual void set_python(PyObject* value_) = 0;
  virtual Ref<> default_python() const = 0;
  virtual Ref<> get_min_python() const = 0;
  virtual Ref<> get_max_python() const = 0;
  virtual void set_allowed_python(PyObject* values) = 0;
  virtual Ref<> allowed_python() const = 0;
  virtual void set_min_python(PyObject* v) = 0;
  virtual void set_max_python(PyObject* v) = 0;
  virtual void set_step_python(PyObject* v) = 0;
#endif

  const type_info& type() const {
    return base().type();
  }

  const string& name_() const { // Can't be named name due to ambiguity in gcc 4.6
    return base().name();
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

  GEODE_CORE_EXPORT void dump(int indent) const ;
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
  GEODE_CORE_EXPORT PropClamp();
public:
  GEODE_CORE_EXPORT ~PropClamp();

  Prop<T>& self() {
    return static_cast<Prop<T>&>(*this);
  }

  T clamp(T value) const {
    return geode::clamp(value,min,max);
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

#ifdef GEODE_PYTHON
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

  GEODE_CORE_EXPORT Prop<T>& set_min(const PropRef<T> p, real alpha = 1);
  GEODE_CORE_EXPORT Prop<T>& set_max(const PropRef<T> p, real alpha = 1);

  Prop<T>& copy_range_from(const PropClamp& p) {
    set_min(p.min);
    set_max(p.max);
    return self();
  }

private:
  void minimize();
  void maximize();
};

template<class T> class GEODE_CORE_CLASS_EXPORT Prop : public Value<T>, public PropBase, public PropClamp<T,has_clamp<T>::value>
{
public:
  GEODE_NEW_FRIEND
  typedef Value<T> Base;
  friend class PropManager;
  friend struct PropClamp<T,has_clamp<T>::value>;
  typedef PropClamp<T,has_clamp<T>::value> Clamp;
  using Base::name;

protected:
  Prop(string const& name, const T& value_)
    : Value<T>(name), PropBase(), default_(value_)
  {
    this->set_value(value_);
  }

  // Properties never go invalid, so update should never be called
  void update() const {
    GEODE_FATAL_ERROR();
  }
public:

  const T default_;
  vector<T> allowed;

  void set(const T& value_) {
    if (!Equals<T>::eval(peek(),value_)) {
      if(allowed.size() && !geode::contains(allowed,value_))
        throw ValueError("value not in allowed values for " + name());
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

#ifdef GEODE_PYTHON

  void set_python(PyObject* value_) {
    set(try_from_python<T>(value_));
  }

  void set_allowed_python(PyObject* values){
    set_allowed(try_from_python<vector<T>>(values));
  }

  Ref<> allowed_python() const {
    return ref(*try_to_python(allowed));
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

  Ref<> default_python() const {
    return ref(*try_to_python(default_));
  }

  Ref<> get_min_python() const {
    if (has_clamp<T>::value)
      return ref(*try_to_python(dynamic_cast<const PropClamp<T,true>*>(this)->min));
    else
      throw ValueError(format("non-clampable prop does not have a min"));
  }

  Ref<> get_max_python() const {
    if (has_clamp<T>::value)
      return ref(*try_to_python(dynamic_cast<const PropClamp<T,true>*>(this)->max));
    else
      throw ValueError(format("non-clampable prop does not have a max"));
  }

#endif

  const ValueBase& base() const {
    return *this;
  }

  // Look at a property without adding a dependency graph node
  const T& peek() const {
    return *static_cast<const T*>(static_cast<const void*>(&this->buffer));
  }

  // WARNING: This will give a mutable reference to the contained prop; to keep sanity in the
  // dependency graph, you will have to call signal.  If you change the mutable reference
  // and an exception prevents you from calling signal, an obscure bug will result.  Use at
  // your own risk.
  T& mutate() {
    return *static_cast<T*>(static_cast<void*>(&this->buffer));
  }

  bool same_default(PropBase& other_) const {
    Prop* other = other_.cast<T>();
    return other && Equals<T>::eval(default_,other->default_);
  }

  string value_str(bool use_default) const {
    return str(use_default ? default_ : this->peek());
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
  typedef typename remove_const<T>::type type;
  typedef typename CopyConst<Prop<type>,T>::type prop_type;
  Ref<typename CopyConst<Prop<type>,T>::type> self;

  PropRef(const string& name, const T& value)
    : self(new_<Prop<T>>(name,value)) {}

  PropRef(typename CopyConst<Prop<type>,T>::type& self)
    : self(geode::ref(self)) {}

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
    PropRef<T> result(self->name(),self->default_);
    result->set_allowed(self->allowed);
    result->copy_range_from(self);
    result->set((*this)());
    return result;
  }
};

template<class T> inline std::ostream& operator<<(std::ostream& output, const PropRef<T>& ref) {
  return output<<ref();
}

#ifdef GEODE_PYTHON

GEODE_CORE_EXPORT PyObject* to_python(const PropBase& prop);
GEODE_CORE_EXPORT PyObject* ptr_to_python(const PropBase* prop);
GEODE_CORE_EXPORT PropBase& prop_from_python(PyObject* object, const type_info& type);
GEODE_CORE_EXPORT Ref<PropBase> make_prop(string const&, PyObject* value);

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
