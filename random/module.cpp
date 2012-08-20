//#####################################################################
// Module Random
//#####################################################################
#include <other/core/python/module.h>
using namespace other;

void wrap_random() {
    OTHER_WRAP(Random)
    OTHER_WRAP(sobol)
    OTHER_WRAP(counter)
}
