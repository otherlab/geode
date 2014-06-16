//#####################################################################
// Class AirPressure
//#####################################################################
//
// Force due to an open or closed air pocket.
//
// For a closed pocket, we assume constant temperature and mass.  The ideal gas law is
//   pV = nRT
// so
//   p = nRT/V
// The work done on the air due to a change of volume dV is
//   dE = -p dV = -nRT/V dV
// which integrates to
//   E = -nRT log V
//
// For an open air "pocket" (touching the atmosphere), both pressure and temperature are constant, giving
//   dE = -p dV
//   E = -pV
//
// Strange behavior will result if the mesh is not closed.
//
//#####################################################################
#pragma once

#include <geode/utility/config.h>
#include <geode/array/Array.h>
#include <geode/force/Force.h>
#include <geode/vector/Vector.h>
#include <geode/mesh/forward.h>
namespace geode {

class AirPressure : public Force<Vector<real,3>> {
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef real T;
  typedef Vector<T,3> TV;
  typedef Force<TV> Base;

  const Ref<TriangleSoup> mesh;
  const bool closed;
  const int side; // +1 for air on the inside of the mesh, -1 for air on the outside
  T temperature; // in Kelvin
  T amount; // in moles; meaningful only if closed
  T pressure; // derived from p = nRT/V if closed, must be specified if open
  bool skip_rotation_terms;
  const T initial_volume;
private:
  Array<Vector<int,3>> local_mesh;
  Array<const TV> X;
  T volume;
  Array<TV> normals; // area weighted times two

protected:
  AirPressure(Ref<TriangleSoup> mesh,Array<const TV> X,bool closed,int side); // defaults to 1 atm
public:
  ~AirPressure();

  void update_position(Array<const TV> X,bool definite);
  void add_frequency_squared(RawArray<T> frequency_squared) const;
  T elastic_energy() const;
  void add_elastic_force(RawArray<TV> F) const;
  void add_elastic_differential(RawArray<TV> dF,RawArray<const TV> dX) const;
  void add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,3> > dFdX) const;
  T damping_energy(RawArray<const TV> V) const;
  void add_damping_force(RawArray<TV> F,RawArray<const TV> V) const;
  T strain_rate(RawArray<const TV> V) const;

  int nodes() const;
  void structure(SolidMatrixStructure& structure) const;
  void add_elastic_gradient(SolidMatrix<TV>& matrix) const;
  void add_damping_gradient(SolidMatrix<TV>& matrix) const;
};
}
