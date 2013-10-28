//#####################################################################
// Module Forces
//#####################################################################
#include <geode/python/wrap.h>
using namespace geode;

void wrap_force() {
  GEODE_WRAP(Force)
  GEODE_WRAP(gravity)
  GEODE_WRAP(springs)
  GEODE_WRAP(finite_volume)
  GEODE_WRAP(strain_measure)
  GEODE_WRAP(strain_measure_hex)
  GEODE_WRAP(constitutive_model)
  GEODE_WRAP(neo_hookean)
  GEODE_WRAP(linear_bending)
  GEODE_WRAP(linear_finite_volume)
  GEODE_WRAP(linear_finite_volume_hex)
  GEODE_WRAP(cubic_hinges)
  GEODE_WRAP(ether_drag)
  GEODE_WRAP(air_pressure)
  GEODE_WRAP(simple_shell)
  GEODE_WRAP(pins)
  GEODE_WRAP(axis_pins)
  GEODE_WRAP(surface_pins)
  GEODE_WRAP(binding_springs)
  GEODE_WRAP(particle_binding_springs)
}
