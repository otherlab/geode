// Python needs numeric_limits

#include <limits>
#include <geode/python/Class.h>
#include <geode/python/numpy.h>
#include <geode/python/wrap.h>
namespace geode {

using std::numeric_limits;

#ifdef GEODE_PYTHON

namespace {
template<class T> struct Limits : public Object {
 private:
  static const char* get_typename();
  static string repr_helper(const T x);
 public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  static const T min, max, epsilon, round_error, infinity, quiet_NaN, signaling_NaN, denorm_min;
  static const int digits = numeric_limits<T>::digits;
  static const int digits10 = numeric_limits<T>::digits10;
  static const int min_exponent = numeric_limits<T>::min_exponent;
  static const int min_exponent10 = numeric_limits<T>::min_exponent10;
  static const int max_exponent = numeric_limits<T>::max_exponent;
  static const int max_exponent10 = numeric_limits<T>::max_exponent10;
  string repr() const {
    // Use separate format calls since Windows lacks variadic templates
    return format("numeric_limits<%s>(\n  min = %s\n  max = %s\n  epsilon = %s\n  round_error = %s\n  quiet_NaN = %s\n",
                  get_typename(),
                  repr_helper(min),repr_helper(max),repr_helper(epsilon),repr_helper(round_error),repr_helper(quiet_NaN))
         + format("  signaling_NaN = %s\n  denorm_min = %s\n",
                  repr_helper(signaling_NaN),repr_helper(denorm_min))
         + format("  digits = %d\n  digits10 = %d\n  min_exponent = %d\n  min_exponent10 = %d\n  max_exponent = %d\n  max_exponent10 = %d)",
                  digits,digits10,min_exponent,min_exponent10,max_exponent,max_exponent10);
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
#undef VALUE
#define COUNT(name) template<class T> const int Limits<T>::name;
COUNT(digits)
COUNT(digits10)
COUNT(min_exponent)
COUNT(min_exponent10)
COUNT(max_exponent)
COUNT(max_exponent10)
#undef COUNT

#define INSTANTIATE(T,fmt) \
  template<> GEODE_DEFINE_TYPE(Limits<T>) \
  template<> const char* Limits<T>::get_typename() { return #T; } \
  template<> string Limits<T>::repr_helper(const T x) { return format("%" fmt,x); }

INSTANTIATE(int32_t,PRId32)
INSTANTIATE(int64_t,PRId64)
INSTANTIATE(uint32_t,PRIu32)
INSTANTIATE(uint64_t,PRIu64)
INSTANTIATE(float,"g")
INSTANTIATE(double,"g")
}

static PyObject* build_limits(PyObject* object) {
  PyArray_Descr* dtype;
  if (!PyArray_DescrConverter(object,&dtype))
    return 0;
  const Ref<> save = steal_ref(*(PyObject*)dtype);
  const int type = dtype->type_num;
  switch (type) {
    #define CASE(T) case NumpyScalar<T>::value: return to_python(new_<Limits<T>>());
    CASE(int32_t)
    CASE(int64_t)
    CASE(uint32_t)
    CASE(uint64_t)
    CASE(float)
    CASE(double)
    default:
      Ref<PyObject> s = steal_ref_check(PyObject_Str((PyObject*)dtype));
      throw TypeError(format("numeric_limits unimplemented for type %s",from_python<const char*>(s)));
  }
}

#endif
}
using namespace geode;

#ifdef GEODE_PYTHON
template<class T> static void wrap_helper() {
  typedef Limits<T> Self;
  Class<Self>("numeric_limits")
    .GEODE_FIELD(min)
    .GEODE_FIELD(max)
    .GEODE_FIELD(epsilon)
    .GEODE_FIELD(round_error)
    .GEODE_FIELD(infinity)
    .GEODE_FIELD(quiet_NaN)
    .GEODE_FIELD(signaling_NaN)
    .GEODE_FIELD(denorm_min)
    .GEODE_FIELD(digits)
    .GEODE_FIELD(digits10)
    .GEODE_FIELD(min_exponent)
    .GEODE_FIELD(min_exponent10)
    .GEODE_FIELD(max_exponent)
    .GEODE_FIELD(max_exponent10)
    .GEODE_REPR()
    ;
}
#endif

void wrap_numeric_limits() {
#ifdef GEODE_PYTHON
  wrap_helper<int32_t>();
  wrap_helper<int64_t>();
  wrap_helper<uint32_t>();
  wrap_helper<uint64_t>();
  wrap_helper<float>();
  wrap_helper<double>();
  GEODE_FUNCTION_2(numeric_limits,build_limits)
#endif
}
