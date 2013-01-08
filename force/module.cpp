//#####################################################################
// Module Forces
//#####################################################################
#include <other/core/python/wrap.h>
using namespace other;

void wrap_force() {
  OTHER_WRAP(Force)
  OTHER_WRAP(gravity)
  OTHER_WRAP(springs)
  OTHER_WRAP(finite_volume)
  OTHER_WRAP(strain_measure)
  OTHER_WRAP(strain_measure_hex)
  OTHER_WRAP(constitutive_model)
  OTHER_WRAP(neo_hookean)
  OTHER_WRAP(linear_bending)
  OTHER_WRAP(linear_finite_volume)
  OTHER_WRAP(linear_finite_volume_hex)
  OTHER_WRAP(ether_drag)
}
