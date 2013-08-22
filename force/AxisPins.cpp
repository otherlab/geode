//#####################################################################
// Class AxisPins
//#####################################################################
#include <other/core/force/AxisPins.h>
#include <other/core/array/NdArray.h>
#include <other/core/array/ProjectedArray.h>
#include <other/core/structure/Hashtable.h>
#include <other/core/python/Class.h>
#include <other/core/vector/SolidMatrix.h>
#include <other/core/vector/SymmetricMatrix.h>
namespace other {

typedef real T;
typedef Vector<T,3> TV;
OTHER_DEFINE_TYPE(AxisPins)

AxisPins::
AxisPins(Array<const int> particles, Array<const T> mass, Array<const TV> targets, NdArray<const T> stiffness, NdArray<const T> damping_ratio)
  : particles(particles)
  , targets(targets)
  , mass(mass)
  , k(particles.size(),false)
  , kd(particles.size(),false)
{
  max_node = particles.size()?particles.max()+1:0;
  OTHER_ASSERT(mass.size()>=max_node);
  OTHER_ASSERT(particles.size()==targets.size());
  OTHER_ASSERT(stiffness.rank()==0 || (stiffness.rank()==1 && stiffness.shape[0]==particles.size()));
  OTHER_ASSERT(damping_ratio.rank()==0 || (damping_ratio.rank()==1 && damping_ratio.shape[0]==particles.size()));

  for (int i=0;i<particles.size();i++) {
    int p = particles[i]; 
    T stiffness_ = stiffness.rank()?stiffness[i]:stiffness();
    T damping_ratio_ = damping_ratio.rank()?damping_ratio[i]:damping_ratio();
    k[i] = stiffness_*mass[p];
    kd[i] = 2*damping_ratio_*mass[p]*sqrt(stiffness_);
  }
}

AxisPins::~AxisPins() {}

int AxisPins::nodes() const {
  return max_node;
}

// Hessian has diagonal terms only, so nothing to do
void AxisPins::structure(SolidMatrixStructure& structure) const {}

void AxisPins::update_position(Array<const TV> X_, bool definite) {
  OTHER_ASSERT(X_.size()==mass.size());
  X = X_;
  axis.normalize();
}

void AxisPins::add_frequency_squared(RawArray<T> frequency_squared) const {
  OTHER_ASSERT(frequency_squared.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    frequency_squared[p] += k[i]/mass[p];
  }
}

T AxisPins::elastic_energy() const {
  T energy = 0;
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    energy += k[i]*sqr(dot(axis,X[p]-targets[i]));
  }
  return energy/2;
}

void AxisPins::add_elastic_force(RawArray<TV> F) const {
  OTHER_ASSERT(F.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    F[p] += k[i]*dot(targets[i]-X[p],axis)*axis;
  }
}

void AxisPins::add_elastic_differential(RawArray<TV> dF, RawArray<const TV> dX) const {
  OTHER_ASSERT(dF.size()==mass.size());
  OTHER_ASSERT(dX.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    dF[p] -= k[i]*dot(dX[p],axis)*axis;
  }
}

void AxisPins::add_elastic_gradient(SolidMatrix<TV>& matrix) const {
  OTHER_ASSERT(matrix.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    matrix.add_entry(p,scaled_outer_product(-k[i],axis));
  }
}

void AxisPins::add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,m>> dFdX) const {
  OTHER_ASSERT(dFdX.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    dFdX[p] -= scaled_outer_product(k[i],axis);
  }
}

T AxisPins::damping_energy(RawArray<const TV> V) const {
  OTHER_ASSERT(V.size()==mass.size());
  T energy=0;
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    energy += kd[i]*sqr(dot(V[p],axis));
  }
  return energy/2;
}

void AxisPins::add_damping_force(RawArray<TV> F,RawArray<const TV> V) const {
  OTHER_ASSERT(V.size()==mass.size());
  OTHER_ASSERT(F.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    F[p] -= kd[i]*dot(V[p],axis)*axis;
  }
}

void AxisPins::add_damping_gradient(SolidMatrix<TV>& matrix) const {
  OTHER_ASSERT(matrix.size()==mass.size());
  for (int i=0;i<particles.size();i++) {
    int p = particles[i];
    matrix.add_entry(p,scaled_outer_product(-kd[i],axis));
  }
}

T AxisPins::strain_rate(RawArray<const TV> V) const {
  return 0;
}

}
using namespace other;

void wrap_axis_pins() {
  typedef AxisPins Self;
  Class<Self>("AxisPins")
    .OTHER_INIT(Array<const int>,Array<const T>,Array<const TV>,NdArray<const T>,NdArray<const T>)
    .OTHER_FIELD(axis)
    ;
}
