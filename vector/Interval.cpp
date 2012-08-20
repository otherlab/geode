//#####################################################################
// Class Interval
//#####################################################################
#include <other/core/vector/Interval.h>
#include <other/core/python/exceptions.h>
namespace other{

template<class T> OTHER_EXPORT PyObject*
to_python(const Interval<T>& self)
{
    const char* format=boost::is_same<T,float>::value?"ff":"dd";
    return Py_BuildValue(format,self.min,self.max);
}

template<class T> Interval<T> FromPython<Interval<T> >::convert(PyObject* object)
{
    Interval<T> self;
    const char* format=boost::is_same<T,float>::value?"ff":"dd";
    if(PyArg_ParseTuple(object,format,&self.min,&self.max))
        return self;
    throw_python_error();
}

#define INSTANTIATE(T) \
  template PyObject* to_python(const Interval<T>&); \
  template Interval<T> FromPython<Interval<T> >::convert(PyObject*);
INSTANTIATE(float)
INSTANTIATE(double)

}
