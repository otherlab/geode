//#####################################################################
// File interrupts
//#####################################################################
#pragma once

#include <other/core/utility/config.h>
namespace other{

// Check if an interrupt has been posted, and throw an exception if so.
// This function is OpenMP thread safe, but any exceptions thrown must
// be caught if inside a parallel block (e.g., use interrupted() instead).
void check_interrupts() OTHER_EXPORT;

// Check if an interrupt has been posted without throwing an exception.
// Internally, interrupt exceptions will be caught and squelched.
// This function is OpenMP thread safe.
bool interrupted() OTHER_EXPORT;

// Add a function to call from check_interrupts.  It must be safe to call
// this function several times for the same interrupt, and checker
// must be thread safe.
void add_interrupt_checker(void (*checker)()) OTHER_EXPORT;

#ifdef OTHER_PYTHON
// Interrupt checker for Python exceptions.  Use only from core/module.cpp.
void check_python_interrupts();
#endif

}
