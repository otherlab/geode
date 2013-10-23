//#####################################################################
// Function from_python
//#####################################################################
//
// Conversion from PyObject* to C++ types.
//
// from_python can be extended by specializing the FromPython class template (since functions cannot be partially specialized), and
// defining a Convert function.  The Convert function takes a borrowed reference to a PyObject, returns the desired type if possible,
// and throws an exception otherwise (preferably via throw_type_error to reduce code bloat).
//
// from_python does not need to be specialized for classes which are native python objects, since predefined conversions to T&, Ref<T>,
// and Ptr<T> are already defined.
//
// from_python<T> is allowed to return a slightly different type than T (e.g., T instead of const T&, or Array<T> instead of RawArray<T>).
//
//#####################################################################
#pragma once

#include <geode/python/config.h>
#include <geode/python/exceptions.h>
#include <geode/python/Object.h>
#include <geode/utility/config.h>
#include <geode/utility/validity.h>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/utility/enable_if.hpp>
#include <string>
namespace geode {

using std::string;

template<class T> static inline auto from_python(PyObject* object)
  -> decltype(FromPython<T>::convert(object)) {
  return FromPython<T>::convert(object);
}

template<class T> static inline auto from_python(const Ref<>& object)
  -> decltype(FromPython<T>::convert(&*object)) {
  return FromPython<T>::convert(&*object);
}

template<> struct FromPython<PyObject*>{static PyObject* convert(PyObject* object) {
  return object;
}};

#ifdef GEODE_PYTHON

template<> struct FromPython<bool>{GEODE_CORE_EXPORT static bool convert(PyObject* object);};
template<> struct FromPython<int>{GEODE_CORE_EXPORT static int convert(PyObject* object);};
template<> struct FromPython<unsigned int>{GEODE_CORE_EXPORT static int convert(PyObject* object);};
template<> struct FromPython<long>{GEODE_CORE_EXPORT static long convert(PyObject* object);};
template<> struct FromPython<unsigned long>{GEODE_CORE_EXPORT static unsigned long convert(PyObject* object);};
template<> struct FromPython<long long>{GEODE_CORE_EXPORT static long long convert(PyObject* object);};
template<> struct FromPython<unsigned long long>{GEODE_CORE_EXPORT static unsigned long long convert(PyObject* object);};
template<> struct FromPython<float>{GEODE_CORE_EXPORT static float convert(PyObject* object);};
template<> struct FromPython<double>{GEODE_CORE_EXPORT static double convert(PyObject* object);};
template<> struct FromPython<const char*>{GEODE_CORE_EXPORT static const char* convert(PyObject* object);};
template<> struct FromPython<string>{GEODE_CORE_EXPORT static string convert(PyObject*);};
template<> struct FromPython<char>{GEODE_CORE_EXPORT static char convert(PyObject* object);};

// Conversion to T& for python types
template<class T> struct FromPython<T&,typename boost::enable_if<boost::is_base_of<Object,T>>::type>{static T&
convert(PyObject* object) {
  if (!boost::is_same<T,Object>::value && &T::pytype==&T::Base::pytype)
    unregistered_python_type(object,&T::pytype,GEODE_DEBUG_FUNCTION_NAME);
  if (!PyObject_IsInstance(object,(PyObject*)&T::pytype))
    throw_type_error(object,&T::pytype);
  return *(T*)(object+1);
}};

#ifdef GEODE_PYTHON
// Declare has_from_python<T>
GEODE_VALIDITY_CHECKER(has_from_python,T,from_python<T>(0))
template<> struct has_from_python<void> : public mpl::true_ {};
#endif

// Conversion to const T& for copy constructible types
template<class T> struct FromPython<const T&,typename boost::enable_if<has_from_python<T>>::type> : public FromPython<T>{};

// Conversion from enums
template<class E> struct FromPython<E,typename boost::enable_if<boost::is_enum<E>>::type> { GEODE_CORE_EXPORT static E convert(PyObject*); };

#endif

// Conversion to const T
template<class T> struct FromPython<const T> : public FromPython<T>{};

}
