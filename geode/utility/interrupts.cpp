//#####################################################################
// File interrupts
//#####################################################################
#include <geode/utility/interrupts.h>
#include <geode/python/exceptions.h>
#include <geode/python/config.h>
#include <vector>
namespace geode {

static std::vector<void(*)()> interrupt_checkers;

void check_interrupts() {
  for (const auto checker : interrupt_checkers)
    checker();
}

bool interrupted() {
  try {
    check_interrupts();
    return false;
  } catch (...) {
    return true;
  }
}

void add_interrupt_checker(void (*checker)()) {
  interrupt_checkers.push_back(checker);
}

#ifdef GEODE_PYTHON
void check_python_interrupts() {
  bool error = false;
  #pragma omp critical
  {
    if (PyErr_Occurred() || PyErr_CheckSignals())
      error = true;
  }
  if (error)
    throw_python_error();
}
#endif

}
