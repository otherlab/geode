//#####################################################################
// Class Pins
//#####################################################################
#include <geode/force/Pins.h>
#include <geode/array/NdArray.h>
#include <geode/array/ProjectedArray.h>
#include <geode/structure/Hashtable.h>
#include <geode/vector/SolidMatrix.h>
#include <geode/vector/SymmetricMatrix.h>
namespace geode {

typedef real T;
typedef Vector<T,3> TV;

Pins::Pins(Array<const int> particles, Array<const T> mass, Array<const TV> targets, NdArray<const T> stiffness, NdArray<const T> damping_ratio)
  : particles(particles)
  , targets(targets)
  , mass(mass)
  , k(particles.size(),uninit)
  , kd(particles.size(),uninit)
{
  max_node = particles.size()?particles.max()+1:0;
  GEODE_ASSERT(mass.size()>=max_node);
  GEODE_ASSERT(particles.size()==targets.size());
  GEODE_ASSERT(stiffness.rank()==0 || (stiffness.rank()==1 && stiffness.shape[0]==particles.size()));
  GEODE_ASSERT(damping_ratio.rank()==0 || (damping_ratio.rank()==1 && damping_ratio.shape[0]==particles.size()));

  for (int i=0;i<particles.size();i++) {
    int p = particles[i]; 
    T stiffness_ = stiffness.rank()?stiffness[i]:stiffness();
    T damping_ratio_ = damping_ratio.rank()?damping_ratio[i]:damping_ratio();
    k[i] = stiffness_*mass[p];
    kd[i] = 2*damping_ratio_*mass[p]*sqrt(stiffness_);
  }
}

Pins::~Pins() {}

int Pins::nodes() const {
  return max_node;
}

// Hessian has diagonal terms only, so nothing to do
void Pins::structure(SolidMatrixStructure& structure) const {}

void Pins::update_position(Array<const TV> X_, bool definite) {
  GEODE_ASSERT(X_.size()==mass.size());
  X = X_;
}

void Pins::add_frequency_squared(RawArray<T> frequency_squared) const {
  GEODE_ASSERT(frequency_squared.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    frequency_squared[p]+=k[i]/mass[p];
  }
}

T Pins::elastic_energy() const {
  T energy = 0;
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    energy += k[i]*sqr_magnitude(X[p]-targets[i]);
  }
  return energy/2;
}

void Pins::add_elastic_force(RawArray<TV> F) const {
  GEODE_ASSERT(F.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    F[p] += k[i]*(targets[i]-X[p]);
  }
}

void Pins::add_elastic_differential(RawArray<TV> dF,RawArray<const TV> dX) const {
  GEODE_ASSERT(dF.size()==mass.size());
  GEODE_ASSERT(dX.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    dF[p] -= k[i]*dX[p];
  }
}

void Pins::add_elastic_gradient(SolidMatrix<TV>& matrix) const {
  GEODE_ASSERT(matrix.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    matrix.add_entry(p,-k[i]);
  }
}

void Pins::add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,m>> dFdX) const {
  GEODE_ASSERT(dFdX.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    dFdX[p] -= k[i];
  }
}

T Pins::damping_energy(RawArray<const TV> V) const {
  GEODE_ASSERT(V.size()==mass.size());
  T energy = 0;
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    energy += kd[i]*sqr_magnitude(V[p]);
  }
  return energy/2;
}

void Pins::add_damping_force(RawArray<TV> F, RawArray<const TV> V) const {
  GEODE_ASSERT(V.size()==mass.size());
  GEODE_ASSERT(F.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    F[p] -= kd[i]*V[p];
  }
}

void Pins::add_damping_gradient(SolidMatrix<TV>& matrix) const {
  GEODE_ASSERT(matrix.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    matrix.add_entry(p,-kd[i]);
  }
}

T Pins::strain_rate(RawArray<const TV> V) const {
  return 0;
}

}
