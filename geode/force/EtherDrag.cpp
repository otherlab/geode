//#####################################################################
// Class EtherDrag
//#####################################################################
#include <geode/force/EtherDrag.h>
#include <geode/vector/SolidMatrix.h>
#include <geode/vector/SymmetricMatrix.h>
namespace geode {

typedef real T;

template<class TV> EtherDrag<TV>::
EtherDrag(Array<const T> mass, T drag)
  : drag(drag), mass(mass) {}

template<class TV> EtherDrag<TV>::
~EtherDrag() {}

template<class TV> int EtherDrag<TV>::nodes() const {
  return mass.size();
}

template<class TV> void EtherDrag<TV>::
structure(SolidMatrixStructure& structure) const {}

template<class TV> void EtherDrag<TV>::
update_position(Array<const TV> X_,bool definite) {}

template<class TV> void EtherDrag<TV>::
add_frequency_squared(RawArray<T> frequency_squared) const {}

template<class TV> T EtherDrag<TV>::
elastic_energy() const {
  return 0;
}

template<class TV> void EtherDrag<TV>::
add_elastic_force(RawArray<TV> F) const {}

template<class TV> void EtherDrag<TV>::
add_elastic_differential(RawArray<TV> dF,RawArray<const TV> dX) const {}

template<class TV> void EtherDrag<TV>::
add_elastic_gradient(SolidMatrix<TV>& matrix) const {}

template<class TV> void EtherDrag<TV>::
add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,m> > dFdX) const {}

template<class TV> T EtherDrag<TV>::
damping_energy(RawArray<const TV> V) const {
  GEODE_ASSERT(mass.size()==V.size());
  T sum = 0;
  if (drag)
    for (int i=0;i<V.size();i++)
      sum += mass[i]*sqr_magnitude(V[i]);
  return (T).5*drag*sum;
}

template<class TV> void EtherDrag<TV>::
add_damping_force(RawArray<TV> force, RawArray<const TV> V) const {
  GEODE_ASSERT(V.size()==mass.size());
  GEODE_ASSERT(force.size()==mass.size());
  if (drag)
    for (int i=0;i<V.size();i++)
      force[i] -= drag*mass[i]*V[i];
}

template<class TV> void EtherDrag<TV>::
add_damping_gradient(SolidMatrix<TV>& matrix) const {
  GEODE_ASSERT(matrix.size()==mass.size());
  if (drag)
    for (int i=0;i<mass.size();i++)
      matrix.add_entry(i,-drag*mass[i]);
}

template<class TV> T EtherDrag<TV>::
strain_rate(RawArray<const TV> V) const {
  return 0;
}

}
