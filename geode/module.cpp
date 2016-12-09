//#####################################################################
// Module geode_wrap
//#####################################################################
#include <geode/python/module.h>
#include <geode/python/wrap.h>
#include <geode/utility/interrupts.h>
using namespace geode;

static bool has_exact() {
#ifdef GEODE_GMP
  return true;
#else
  return false;
#endif
}

GEODE_PYTHON_MODULE(geode_wrap) {
  GEODE_WRAP(python)
  GEODE_WRAP(utility)

#ifdef GEODE_PYTHON
  // Check for python exceptions in check_interrupts()
  add_interrupt_checker(check_python_interrupts);
#endif

  GEODE_WRAP(math)
  GEODE_WRAP(array)
  GEODE_WRAP(vector)
  GEODE_WRAP(geometry)
  GEODE_WRAP(image)
  GEODE_WRAP(mesh)
  GEODE_WRAP(random)
  GEODE_WRAP(value)
  GEODE_WRAP(force)
  GEODE_WRAP(solver)
  GEODE_WRAP(structure)
  GEODE_WRAP(svg_to_bezier)
#ifdef GEODE_GMP
  GEODE_WRAP(exact)
#endif

  GEODE_FUNCTION(has_exact)
}
