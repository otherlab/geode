// Python needs numeric_limits

#include <limits>
#include <other/core/python/Class.h>
#include <other/core/python/numpy.h>
#include <other/core/python/wrap.h>
namespace other {

using std::numeric_limits;

#ifdef OTHER_PYTHON

namespace {
template<class T> struct Limits : public Object, public numeric_limits<T> {
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
  static const T min, max, epsilon, round_error, infinity, quiet_NaN, signaling_NaN, denorm_min;

  string repr() const {
    // Use separate format calls since Windows lacks variadic templates
    return format("numeric_limits<%s>:\n  min = %g\n  max = %g\n  epsilon = %g\n  round_error = %g\n  quiet_NaN = %g\n",
                  boost::is_same<T,float>::value?"float":"double",
                  min,max,epsilon,round_error,quiet_NaN)
         + format("  signaling_NaN = %g\n  denorm_min = %g\n",
                  signaling_NaN,denorm_min)
         + format("  digits = %d\n  digits10 = %d\n  min_exponent = %d\n  min_exponent10 = %d\n  max_exponent = %d\n  max_exponent10 = %d",
                  this->digits,this->digits10,this->min_exponent,this->min_exponent10,this->max_exponent,this->max_exponent10);
  }
};
#define VALUE(name) template<class T> const T Limits<T>::name = numeric_limits<T>::name();
VALUE(min)
VALUE(max)
VALUE(epsilon)
VALUE(round_error)
VALUE(infinity)
VALUE(quiet_NaN)
VALUE(signaling_NaN)
VALUE(denorm_min)
#define COUNT(name) template<class T> const int Limits<T>::name = numeric_limits<T>::name;

template<> OTHER_DEFINE_TYPE(Limits<float>)
template<> OTHER_DEFINE_TYPE(Limits<double>)
}

static PyObject* build_limits(PyObject* dtype) {
  if (!PyArray_DescrCheck(dtype))
    throw TypeError(format("expected numpy descriptor, got %s",dtype->ob_type->tp_name));
  const int type = ((PyArray_Descr*)dtype)->type_num;
  switch (type) {
    case NumpyScalar<float>::value:  return to_python(new_<Limits<float>>());
    case NumpyScalar<double>::value: return to_python(new_<Limits<double>>());
    default:
      Ref<PyObject> s = steal_ref_check(PyObject_Str(dtype));
      throw TypeError(format("numeric_limits unimplemented for type %s",from_python<const char*>(s)));
  }
}

#endif
}
using namespace other;

#ifdef OTHER_PYTHON
template<class T> static void wrap_helper() {
  typedef Limits<T> Self;
  Class<Self>("numeric_limits")
    .OTHER_FIELD(min)
    .OTHER_FIELD(max)
    .OTHER_FIELD(epsilon)
    .OTHER_FIELD(round_error)
    .OTHER_FIELD(infinity)
    .OTHER_FIELD(quiet_NaN)
    .OTHER_FIELD(signaling_NaN)
    .OTHER_FIELD(denorm_min)
    .OTHER_FIELD(digits)
    .OTHER_FIELD(digits10)
    .OTHER_FIELD(min_exponent)
    .OTHER_FIELD(min_exponent10)
    .OTHER_FIELD(max_exponent)
    .OTHER_FIELD(max_exponent10)
    .OTHER_REPR()
    ;
}
#endif

void wrap_numeric_limits() {
#ifdef OTHER_PYTHON
  wrap_helper<float>();
  wrap_helper<double>();
  OTHER_FUNCTION_2(numeric_limits,build_limits)
#endif
}
