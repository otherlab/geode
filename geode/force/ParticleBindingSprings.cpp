//#####################################################################
// Class ParticleBindingSprings
//#####################################################################
#include <geode/force/ParticleBindingSprings.h>
#include <geode/array/NdArray.h>
#include <geode/array/IndirectArray.h>
#include <geode/array/view.h>
#include <geode/structure/Hashtable.h>
#include <geode/python/Class.h>
#include <geode/vector/SolidMatrix.h>
#include <geode/vector/SymmetricMatrix.h>
namespace geode {

typedef real T;
typedef Vector<T,3> TV;
GEODE_DEFINE_TYPE(ParticleBindingSprings)

ParticleBindingSprings::ParticleBindingSprings(Array<const Vector<int,2>> nodes, Array<const T> mass, NdArray<const T> stiffness, NdArray<const T> damping_ratio)
  : mass(mass)
  , info(nodes.size(),uninit) {
  const int max_node = nodes.size()?scalar_view(nodes).max()+1:0;
  GEODE_ASSERT(mass.size()>=max_node);
  GEODE_ASSERT(stiffness.rank()==0 || (stiffness.rank()==1 && stiffness.shape[0]==nodes.size()));
  GEODE_ASSERT(damping_ratio.rank()==0 || (damping_ratio.rank()==1 && damping_ratio.shape[0]==nodes.size()));

  for (int b=0;b<info.size();b++) {
    ParticleBindingInfo& I = info[b];
    I.nodes = nodes[b];
    T harmonic_mass = 1/(1/mass[I.nodes[0]]+1/mass[I.nodes[1]]);
    T stiffness_ = stiffness.rank()?stiffness[b]:stiffness();
    T damping_ratio_ = damping_ratio.rank()?damping_ratio[b]:damping_ratio();
    I.k = stiffness_*harmonic_mass;
    I.kd = 2*damping_ratio_*harmonic_mass*sqrt(stiffness_);
  }
}

ParticleBindingSprings::~ParticleBindingSprings() {}

int ParticleBindingSprings::nodes() const {
  return mass.size();
}

void ParticleBindingSprings::structure(SolidMatrixStructure& structure) const {
  GEODE_ASSERT(mass.size()==structure.size());
  for (int b=0;b<info.size();b++) {
    const ParticleBindingInfo& I = info[b];
    structure.add_entry(I.nodes[0],I.nodes[1]);
  }
}

void ParticleBindingSprings::update_position(Array<const TV> X_, bool definite) {
  GEODE_ASSERT(X_.size()==mass.size());
  X = X_;
}

void ParticleBindingSprings::add_frequency_squared(RawArray<T> frequency_squared) const {
  // Ignore CFL contribution for now
}

static inline TV gather(RawArray<const TV> X,const ParticleBindingInfo& I) {
  return X[I.nodes[0]]-X[I.nodes[1]];
}

static inline void scatter(RawArray<TV> F,const ParticleBindingInfo& I,const TV& f) {
  F[I.nodes[0]] -= f;
  F[I.nodes[1]] += f;
}

static inline void add_entries(SolidMatrix<TV>& matrix,const ParticleBindingInfo& I,T scale) {
  matrix.add_entry(I.nodes[0],-scale);
  matrix.add_entry(I.nodes[1],-scale);
  matrix.add_entry(I.nodes[0],I.nodes[1],scale);
}

T ParticleBindingSprings::elastic_energy() const {
  T energy = 0;
  for (int b=0;b<info.size();b++) {
    const ParticleBindingInfo& I = info[b];
    energy += I.k*sqr_magnitude(gather(X,I));
  }
  return energy/2;
}

void ParticleBindingSprings::add_elastic_force(RawArray<TV> F) const {
  add_elastic_differential(F,X);
}

void ParticleBindingSprings::add_elastic_differential(RawArray<TV> dF,RawArray<const TV> dX) const {
  GEODE_ASSERT(dF.size()==mass.size());
  GEODE_ASSERT(dX.size()==mass.size());
  for (int b=0;b<info.size();b++) {
    const ParticleBindingInfo& I = info[b];
    scatter(dF,I,I.k*gather(dX,I));
  }
}

void ParticleBindingSprings::add_elastic_gradient(SolidMatrix<TV>& matrix) const {
  GEODE_ASSERT(matrix.size()==mass.size());
  for (int b=0;b<info.size();b++) {
    const ParticleBindingInfo& I = info[b];
    add_entries(matrix,I,I.k);
  }
}

void ParticleBindingSprings::add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,m>> dFdX) const {
  GEODE_ASSERT(dFdX.size()==mass.size());
  for (int b=0;b<info.size();b++) {
    const ParticleBindingInfo& I = info[b];
    dFdX[I.nodes[0]] -= I.k;
    dFdX[I.nodes[1]] -= I.k;
  }
}

T ParticleBindingSprings::damping_energy(RawArray<const TV> V) const {
  GEODE_ASSERT(V.size()==mass.size());
  T energy = 0;
  for (int b=0;b<info.size();b++) {
    const ParticleBindingInfo& I = info[b];
    energy += I.kd*sqr_magnitude(gather(V,I));
  }
  return energy/2;
}

void ParticleBindingSprings::add_damping_force(RawArray<TV> F,RawArray<const TV> V) const {
  GEODE_ASSERT(V.size()==mass.size());
  GEODE_ASSERT(F.size()==mass.size());
  for (int b=0;b<info.size();b++) {
    const ParticleBindingInfo& I = info[b];
    scatter(F,I,I.kd*gather(V,I));
  }
}

void ParticleBindingSprings::add_damping_gradient(SolidMatrix<TV>& matrix) const {
  GEODE_ASSERT(matrix.size()==mass.size());
  for (int b=0;b<info.size();b++) {
    const ParticleBindingInfo& I = info[b];
    add_entries(matrix,I,I.kd);
  }
}

T ParticleBindingSprings::strain_rate(RawArray<const TV> V) const {
  return 0;
}

}
using namespace geode;

void wrap_particle_binding_springs() {
  typedef ParticleBindingSprings Self;
  Class<Self>("ParticleBindingSprings")
    .GEODE_INIT(Array<const Vector<int,2>>,Array<const T>,NdArray<const T>,NdArray<const T>)
    ;
}
