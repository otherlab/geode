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

#include <other/core/python/config.h>
#include <other/core/python/exceptions.h>
#include <other/core/python/Object.h>
#include <other/core/utility/config.h>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/utility/enable_if.hpp>
#include <string>
namespace other {

using std::string;
} namespace boost { template<class T> class shared_ptr;
} namespace other {

using boost::shared_ptr;

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

#ifdef OTHER_PYTHON

template<> struct FromPython<bool>{static bool convert(PyObject* object) OTHER_CORE_EXPORT;};
template<> struct FromPython<int>{static int convert(PyObject* object) OTHER_CORE_EXPORT;};
template<> struct FromPython<unsigned int>{static int convert(PyObject* object) OTHER_CORE_EXPORT;};
template<> struct FromPython<long>{static long convert(PyObject* object) OTHER_CORE_EXPORT;};
template<> struct FromPython<unsigned long>{static unsigned long convert(PyObject* object) OTHER_CORE_EXPORT;};
template<> struct FromPython<unsigned long long>{static unsigned long long convert(PyObject* object) OTHER_CORE_EXPORT;};
template<> struct FromPython<float>{static float convert(PyObject* object) OTHER_CORE_EXPORT;};
template<> struct FromPython<double>{static double convert(PyObject* object) OTHER_CORE_EXPORT;};
template<> struct FromPython<const char*>{static const char* convert(PyObject* object) OTHER_CORE_EXPORT;};
template<> struct FromPython<string>{static string convert(PyObject*) OTHER_CORE_EXPORT;};
template<> struct FromPython<char>{static char convert(PyObject* object) OTHER_CORE_EXPORT;};

// Conversion to T& for python types
template<class T> struct FromPython<T&,typename boost::enable_if<boost::is_base_of<Object,T> >::type>{static T&
convert(PyObject* object) {
  if (!boost::is_same<T,Object>::value && &T::pytype==&T::Base::pytype)
    unregistered_python_type(object,&T::pytype);
  if (!PyObject_IsInstance(object,(PyObject*)&T::pytype))
    throw_type_error(object,&T::pytype);
  return *(T*)(object+1);
}};

// Conversion to const T& for non-python types
template<class T> struct FromPython<const T&,typename boost::disable_if<boost::is_base_of<Object,T> >::type> : public FromPython<T>{};

// Conversion from enums
template<class E> struct FromPython<E,typename boost::enable_if<boost::is_enum<E>>::type> { static E convert(PyObject*) OTHER_CORE_EXPORT; };

#endif

// Conversion to const T
template<class T> struct FromPython<const T> : public FromPython<T>{};

// TODO: Currently, Value<T> requires T to be Python convertible.  Therefore, we pretend shared_ptr is convertible
// to allow arbitrary types to be hacked in.  This should be fixed by removing the restriction from Value<T>.
template<class T> struct FromPython<shared_ptr<T> >{static shared_ptr<T> convert(PyObject* object) {
  OTHER_NOT_IMPLEMENTED();
}};

}
