//#####################################################################
// Class Gravity
//#####################################################################
#include <geode/force/Gravity.h>
#include <geode/python/Class.h>
namespace geode {

typedef real T;
template<> GEODE_DEFINE_TYPE(Gravity<Vector<real,3> >)

template<class TV> Gravity<TV>::Gravity(Array<const T> mass)
  : mass(mass) {
  gravity[m-1] = -9.8;
}

template<class TV> Gravity<TV>::~Gravity() {}

template<class TV> int Gravity<TV>::nodes() const {
  return mass.size();
}

template<class TV> void Gravity<TV>::update_position(Array<const TV> X_,bool definite) {
  X = X_;
  GEODE_ASSERT(X.size()==mass.size());
}

template<class TV> void Gravity<TV>::add_frequency_squared(RawArray<T> frequency_squared) const {}

template<class TV> typename TV::Scalar Gravity<TV>::elastic_energy() const {
  T energy=0;
  for (int i=0;i<mass.size();i++)
    energy -= mass[i]*dot(gravity,X[i]);
  return energy;
}

template<class TV> void Gravity<TV>::add_elastic_force(RawArray<TV> F) const {
  GEODE_ASSERT(F.size()==mass.size());
  for (int i=0;i<mass.size();i++)
      F[i] += mass[i]*gravity;
}

template<class TV> void Gravity<TV>::add_elastic_differential(RawArray<TV> dF,RawArray<const TV> dX) const {}

template<class TV> void Gravity<TV>::add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,m> > dFdX) const {}

template<class TV> typename TV::Scalar Gravity<TV>::damping_energy(RawArray<const TV> V) const {
  return 0;
}

template<class TV> void Gravity<TV>::add_damping_force(RawArray<TV> force,RawArray<const TV> V) const {}

template<class TV> typename TV::Scalar Gravity<TV>::strain_rate(RawArray<const TV> V) const {
  return 0;
}

template<class TV> void Gravity<TV>::structure(SolidMatrixStructure& structure) const {}

template<class TV> void Gravity<TV>::add_elastic_gradient(SolidMatrix<TV>& matrix) const {}

template<class TV> void Gravity<TV>::add_damping_gradient(SolidMatrix<TV>& matrix) const {}

}
using namespace geode;

void wrap_gravity() {
  typedef real T;
  typedef Vector<T,3> TV;
  typedef Gravity<TV> Self;
  Class<Self>("Gravity")
    .GEODE_INIT(Array<const T>)
    .GEODE_FIELD(gravity)
    ;
}
