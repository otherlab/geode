#include <other/core/value/Prop.h>
#include <other/core/value/Listen.h>
#include <other/core/array/NdArray.h>
#include <other/core/python/Class.h>
#include <other/core/python/numpy.h>
#include <other/core/python/Ptr.h>
#include <other/core/structure/Tuple.h>
#include <other/core/utility/const_cast.h>
#include <other/core/utility/curry.h>
#include <other/core/utility/format.h>
#include <other/core/vector/Frame.h>
#include <other/core/vector/Rotation.h>
#include <iostream>
namespace other{

using std::cout;
using std::endl;
using std::exception;
using std::numeric_limits;
typedef real T;
typedef Vector<T,2> TV2;
typedef Vector<T,3> TV3;
typedef Vector<T,4> TV4;

PropBase::PropBase()
  : hidden(false)
  , required(false)
  , abbrev(0)
{}

PropBase::~PropBase() {}

void PropBase::dump(int indent) const {
  printf("%*sProp(\"%s\",%s)\n",2*indent,"",name_().c_str(),value_str().c_str());
}

// Since PropBase doesn't by itself inherit from Object due to multiple inheritance woes,
// we need a special wrapper class to expose Prop<T> to python.

#ifdef OTHER_PYTHON
Ref<PropBase> make_prop(const string& n, PyObject* value) {
  // If the value has known simple type, make the corresponding property
  if (PyBool_Check(value))
    return new_<Prop<bool>>(n,from_python<bool>(value));
  if (PyInt_Check(value))
    return new_<Prop<int>>(n,from_python<int>(value));
  if (PyFloat_Check(value))
    return new_<Prop<double>>(n,from_python<double>(value));
  if (PyString_Check(value))
    return new_<Prop<string>>(n,from_python<string>(value));
  if (PySequence_Check(value)) {
    if (PyArray_Check(value)) {
      if (rotations_check<TV2>(value))
        return new_<Prop<Rotation<TV2>>>(n,from_python<Rotation<TV2>>(value));
      if (rotations_check<TV3>(value))
        return new_<Prop<Rotation<TV3>>>(n,from_python<Rotation<TV3>>(value));
      if (frames_check<TV2>(value))
        return new_<Prop<Frame<TV2>>>(n,from_python<Frame<TV2>>(value));
      if (frames_check<TV3>(value))
        return new_<Prop<Frame<TV3>>>(n,from_python<Frame<TV3>>(value));
    }
    NdArray<const real> a = from_python<NdArray<const real>>(value);
    if (a.shape.size()==1 && a.shape[0]==2)
      return new_<Prop<TV2>>(n,vec(a[0],a[1]));
    if (a.shape.size()==1 && a.shape[0]==3)
      return new_<Prop<TV3>>(n,vec(a[0],a[1],a[2]));
    if (a.shape.size()==1 && a.shape[0]==4)
      return new_<Prop<TV4>>(n,vec(a[0],a[1],a[2],a[3]));
    throw TypeError(format("don't know how to make a vector property of shape %s",str(a.shape)));
  }
  if (value == Py_None) {
    return new_<Prop<Ref<>>>(n, ref(*value));
  }

  // We don't understand the value, so complain
  throw TypeError(format("we don't know how to make a property of type '%s'",value->ob_type->tp_name));
}
#endif

template<class T> PropClamp<T,true>::PropClamp()
  : min(-numeric_limits<T>::max())
  , max( numeric_limits<T>::max())
  , step(max) {}

template<class T> PropClamp<T,true>::~PropClamp() {}

template<class T> Prop<T>& PropClamp<T,true>::set_min(const PropRef<T> p, real alpha) {
  Prop<T>& self = this->self();
  OTHER_ASSERT(p->name != self.name && !(p->prop_min && p->prop_min->x->name == self.name));
  prop_min.reset(new Tuple<PropRef<T>,Ref<Listen>,real>(p,listen(p,curry(&Self::minimize,this)),alpha));
  minimize();
  return self;
}

template<class T> Prop<T>& PropClamp<T,true>::set_max(const PropRef<T> p, real alpha) {
  auto& self = this->self();
  OTHER_ASSERT(p->name != self.name && !(p->prop_max && p->prop_max->x->name == self.name));
  prop_max.reset(new Tuple<PropRef<T>,Ref<Listen>,real>(p,listen(p,curry(&Self::maximize,this)),alpha));
  maximize();
  return self;
}

template<class T> void PropClamp<T,true>::minimize() {
  const auto& self = this->self();
  min = T(prop_min->x()*prop_min->z);
  T v = self();
  self.set_value(v < min ? min : v); //a bit dirty, but we want to trigger update to toolbar
}

template<class T> void PropClamp<T,true>::maximize() {
  const auto& self = this->self();
  max = T(prop_max->x()*prop_max->z);
  T v = self();
  self.set_value(v > max ? max : v); //a bit dirty, but we want to trigger update to toolbar
}

template struct PropClamp<int,true>;
template struct PropClamp<double,true>;

#ifdef OTHER_PYTHON

PyObject* to_python(const PropBase& prop) {
  return to_python(prop.base());
}

PyObject* ptr_to_python(const PropBase* prop) {
  return (PyObject*)&prop->base()-1;
}

PropBase& FromPython<PropBase&>::convert(PyObject* object) {
  ValueBase& value = from_python<ValueBase&>(object);
  if (PropBase* prop = dynamic_cast<PropBase*>(&value))
    return *prop;
  throw TypeError("expected Prop, got Value");
}

PropBase& prop_from_python(PyObject* object, const type_info& goal) {
  PropBase& self = from_python<PropBase&>(object);
  const type_info& type = self.type();
  if (type==goal || !strcmp(type.name(),goal.name()))
    return self;
  throw TypeError(format("expected Prop<%s>, got Prop<%s>",goal.name(),self.type().name()));
}

#endif

// Reduce template bloat
template class Prop<bool>;
template class Prop<int>;
template class Prop<double>;
template class Prop<string>;
template class Prop<TV2>;
template class Prop<TV3>;
template class Prop<TV4>;

}
using namespace other;

void wrap_prop() {
#ifdef OTHER_PYTHON
  OTHER_FUNCTION_2(Prop,make_prop)
#endif
}
