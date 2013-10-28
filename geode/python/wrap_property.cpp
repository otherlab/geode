#ifdef GEODE_PYTHON
#include <geode/python/wrap_property.h>
#include <geode/utility/const_cast.h>
namespace geode {

PyObject* wrap_property_helper(PyTypeObject* type,const char* name,getter get_wrapper,setter set_wrapper,void* getset) {
  // Make a PyGetSetDef
  PyGetSetDef* def = (PyGetSetDef*)malloc(sizeof(PyGetSetDef));
  memset(def,0,sizeof(PyGetSetDef));
  def->name = const_cast_(name);
  def->get = get_wrapper;
  def->set = set_wrapper;
  def->closure = getset;

  // Allocate descriptor
  PyObject* descr = PyDescr_NewGetSet(type,def);
  if (!descr) throw_python_error();
  return descr;
}

}
#endif
