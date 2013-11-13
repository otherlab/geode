#include <geode/value/Prop.h>
#include <geode/value/Listen.h>
#include <geode/array/NdArray.h>
#include <geode/array/view.h>
#include <geode/python/Class.h>
#include <geode/python/numpy.h>
#include <geode/python/Ptr.h>
#include <geode/structure/Tuple.h>
#include <geode/utility/const_cast.h>
#include <geode/utility/curry.h>
#include <geode/utility/format.h>
#include <geode/vector/Frame.h>
#include <geode/vector/Rotation.h>
#include <iostream>
#include <stdio.h>
namespace geode {

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
  printf("%*sProp('%s',%s)\n",2*indent,"",name_().c_str(),value_str().c_str());
}

// Since PropBase doesn't by itself inherit from Object due to multiple inheritance woes,
// we need a special wrapper class to expose Prop<T> to python.

#ifdef GEODE_PYTHON
Ref<PropBase> make_prop_shape(const string& n, NdArray<const T> a, RawArray<const int> shape) {
  const int rank = a.rank();
  GEODE_ASSERT(rank==shape.size());
  const int fixed = shape.count_matches(-1);
  if (shape.slice(0,fixed).count_matches(-1)!=fixed)
    throw ValueError(format("Prop: -1's in shape must occur at the beginning, got %s",str(shape)));
  for (int i=fixed;i<rank;i++)
    if (a.shape[i]!=shape[i])
      throw ValueError(format("Prop: default shape %s does not match shape specification %s",str(a.shape),str(shape)));
  if (rank==1 && fixed==0) {
    if (shape[0]==2) return new_<Prop<TV2>>(n,vec(a[0],a[1]));
    if (shape[0]==3) return new_<Prop<TV3>>(n,vec(a[0],a[1],a[2]));
    if (shape[0]==4) return new_<Prop<TV4>>(n,vec(a[0],a[1],a[2],a[3]));
  } else if (rank==1 && fixed==1)
    return new_<Prop<Array<const T>>>(n,a.flat);
  else if (rank==2 && fixed==1) {
    if (shape[1]==2) return new_<Prop<Array<const TV2>>>(n,vector_view_own<2>(a.flat));
    if (shape[1]==3) return new_<Prop<Array<const TV3>>>(n,vector_view_own<3>(a.flat));
    if (shape[1]==4) return new_<Prop<Array<const TV4>>>(n,vector_view_own<4>(a.flat));
  }
  throw NotImplementedError(format("Prop: shape specification %s is not implemented",str(shape)));
}

Ref<PropBase> make_prop(const string& n, PyObject* value) {
  // If the value has known simple type, make the corresponding property
  if (PyBool_Check(value))
    return new_<Prop<bool>>(n,from_python<bool>(value));
  if (PyInt_Check(value))
    return new_<Prop<int>>(n,from_python<int>(value));
  if (PyFloat_Check(value))
    return new_<Prop<double>>(n,from_python<double>(value));
  if (PyString_Check(value) || PyUnicode_Check(value))
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
    NdArray<const T> a;
    try {
      a = from_python<NdArray<const T>>(value);
    } catch (const exception&) {
      // If the NdArray conversion fails, squelch the error and fall back to our default
      PyErr_Clear();
    }
    return make_prop_shape(n,a,a.shape);
  }

  // Default to a property containing an arbitrary python object
  return new_<Prop<Ref<>>>(n,ref(*value));
}
#endif

template<class T> PropClamp<T,true>::PropClamp()
  : min(-numeric_limits<T>::max())
  , max( numeric_limits<T>::max())
  , step(max) {}

template<class T> PropClamp<T,true>::~PropClamp() {}

template<class T> Prop<T>& PropClamp<T,true>::set_min(const PropRef<T> p, real alpha) {
  Prop<T>& self = this->self();
  GEODE_ASSERT(p->name != self.name && !(p->prop_min && p->prop_min->x->name == self.name));
  prop_min.reset(new Tuple<PropRef<T>,Ref<Listen>,real>(p,listen(p,curry(&Self::minimize,this)),alpha));
  minimize();
  return self;
}

template<class T> Prop<T>& PropClamp<T,true>::set_max(const PropRef<T> p, real alpha) {
  auto& self = this->self();
  GEODE_ASSERT(p->name != self.name && !(p->prop_max && p->prop_max->x->name == self.name));
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

#ifdef GEODE_PYTHON

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

namespace {
struct Unusable {
  bool operator==(Unusable) const { return true; }
  friend ostream& operator<<(ostream& output, Unusable) { return output; }
};
}

// A Prop with a non-python convertible type
static Ref<PropBase> unusable_prop_test() {
  BOOST_MPL_ASSERT((has_to_python<int>));
  return new_<Prop<Unusable>>("unusable",Unusable());
}

#endif

}
using namespace geode;

void wrap_prop() {
#ifdef GEODE_PYTHON
  GEODE_FUNCTION(make_prop)
  GEODE_FUNCTION(make_prop_shape)
  GEODE_FUNCTION(unusable_prop_test)
#endif
}
