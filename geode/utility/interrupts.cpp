// Check for interrupts

#include <geode/utility/interrupts.h>
#include <geode/utility/exceptions.h>
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

}
